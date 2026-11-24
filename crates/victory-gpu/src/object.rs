//! What every console graphics object starts with: a function table
//! pointer, bookkeeping bits and a reference count. Tables begin with
//! IUnknown's three slots. Our objects are static and never destroyed.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

pub type HRESULT = i32;
pub const S_OK: HRESULT = 0;
pub(crate) const S_FALSE: HRESULT = 1;
pub(crate) const E_NOTIMPL: HRESULT = 0x8000_4001_u32 as i32;
pub(crate) const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;
pub(crate) const E_POINTER: HRESULT = 0x8000_4003_u32 as i32;
pub(crate) const E_INVALIDARG: HRESULT = 0x8007_0057_u32 as i32;
pub(crate) const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;
/// What D3D and DXGI return for a missing item (adapter, output, private
/// data).
pub(crate) const NOT_FOUND: HRESULT = 0x887A_0002_u32 as i32;
pub(crate) const MORE_DATA: HRESULT = 0x887A_0003_u32 as i32;

pub type Iid = [u8; 16];

/// An interface id from its text form's fields, in memory order.
pub const fn guid(a: u32, b: u16, c: u16, d: [u8; 8]) -> Iid {
    let (a, b, c) = (a.to_le_bytes(), b.to_le_bytes(), c.to_le_bytes());
    [
        a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], d[0], d[1], d[2], d[3], d[4], d[5], d[6],
        d[7],
    ]
}

pub fn guid_text(g: &Iid) -> String {
    let a = u32::from_le_bytes([g[0], g[1], g[2], g[3]]);
    let b = u16::from_le_bytes([g[4], g[5]]);
    let c = u16::from_le_bytes([g[6], g[7]]);
    let d: String = g[8..].iter().map(|x| format!("{x:02x}")).collect();
    format!("{a:08x}-{b:04x}-{c:04x}-{}-{}", &d[..4], &d[4..])
}

#[repr(C)]
pub struct Object {
    functions: *const *const (),
    /// Device index (bits 0-2) and bookkeeping bits; all zero.
    bits: u32,
    refs: AtomicU32,
}
// SAFETY: the table is static and immutable; the count is atomic.
unsafe impl Sync for Object {}

impl Object {
    pub(crate) const fn new<const N: usize>(table: &'static Table<N>) -> Self {
        Object {
            functions: table.0.as_ptr(),
            bits: 0,
            refs: AtomicU32::new(0),
        }
    }
}

pub(crate) type This = *const Object;

#[repr(transparent)]
pub(crate) struct Table<const N: usize>(pub [*const (); N]);
// SAFETY: function addresses only.
unsafe impl<const N: usize> Sync for Table<N> {}

pub(crate) extern "system" fn add_ref(this: This) -> u32 {
    // SAFETY: `this` is one of our objects.
    let refs = &unsafe { &*this }.refs;
    // COM's AddRef never returns 0, so saturate rather than wrap: a count
    // that wrapped would make the object look unreferenced and freeable.
    let current =
        match refs.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1)) {
            Ok(n) | Err(n) => n,
        };
    current.saturating_add(1)
}

pub(crate) extern "system" fn release(this: This) -> u32 {
    // SAFETY: `this` is one of our objects.
    let refs = &unsafe { &*this }.refs;
    let _ = refs.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1));
    refs.load(Ordering::Relaxed)
}

/// Answers an interface query (or any call handing out an object): a new
/// reference to `found`, or null and `E_NOINTERFACE`.
pub(crate) fn answer(kind: usize, iid: *const Iid, found: Option<This>, out: *mut This) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    let hr = match found {
        Some(object) => {
            add_ref(object);
            // SAFETY: checked non-null; the title passes room for a pointer.
            unsafe { out.write(object) };
            return S_OK;
        }
        None => E_NOINTERFACE,
    };
    // SAFETY: the title passes an interface id (null is tolerated).
    let iid = unsafe { iid.as_ref() }.map(guid_text).unwrap_or_default();
    tracing::warn!(object = KINDS[kind], %iid, "unknown interface");
    // SAFETY: checked non-null.
    unsafe { out.write(std::ptr::null()) };
    hr
}

/// `IUnknown`, which every COM object answers for.
pub(crate) const IUNKNOWN: Iid = guid(0, 0, 0, [0xc0, 0, 0, 0, 0, 0, 0, 0x46]);

/// Whether `iid` is one of `known`, or `IUnknown`.
pub(crate) fn is(iid: *const Iid, known: &[Iid]) -> bool {
    // SAFETY: the title passes an interface id (null is tolerated).
    unsafe { iid.as_ref() }.is_some_and(|i| *i == IUNKNOWN || known.contains(i))
}

/// Hands a new object to the title as one of `known`, or drops it. `T`
/// starts with its `Object`; its table's release slot is
/// [`release_boxed::<T>`].
pub(crate) fn hand_out<T>(
    kind: usize,
    object: Box<T>,
    iid: *const Iid,
    known: &[Iid],
    out: *mut This,
) -> HRESULT {
    if !is(iid, known) {
        return answer(kind, iid, None, out);
    }
    if out.is_null() {
        // Asked only whether creation would succeed.
        return S_FALSE;
    }
    answer(kind, iid, Some(Box::into_raw(object).cast()), out)
}

/// Release for objects made by [`hand_out`]: the last reference frees it.
pub(crate) fn release_boxed<T>(this: This) -> u32 {
    // SAFETY: `this` is the header of a boxed `T`.
    let refs = &unsafe { &*this }.refs;
    match refs.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1)) {
        Ok(1) => {
            private_data().retain(|&(object, _), _| object != this as usize);
            // SAFETY: that was the last reference to a `Box<T>`.
            drop(unsafe { Box::from_raw(this.cast_mut().cast::<T>()) });
            0
        }
        Ok(n) => n - 1,
        Err(_) => 0,
    }
}

/// Private data of every object, by object address and id.
static PRIVATE: Mutex<BTreeMap<(usize, Iid), Vec<u8>>> = Mutex::new(BTreeMap::new());

fn private_data() -> std::sync::MutexGuard<'static, BTreeMap<(usize, Iid), Vec<u8>>> {
    PRIVATE.lock().unwrap_or_else(|e| e.into_inner())
}

extern "system" fn get_private_data(
    this: This,
    id: *const Iid,
    size: *mut u32,
    data: *mut u8,
) -> HRESULT {
    // SAFETY: the title passes an id and its buffer size (null is tolerated).
    let (Some(id), Some(size)) = (unsafe { (id.as_ref(), size.as_mut()) }) else {
        return E_INVALIDARG;
    };
    let map = private_data();
    let Some(value) = map.get(&(this as usize, *id)) else {
        *size = 0;
        return NOT_FOUND;
    };
    let len = value.len() as u32;
    let room = std::mem::replace(size, len);
    if data.is_null() {
        return S_OK;
    }
    if room < len {
        return MORE_DATA;
    }
    // SAFETY: the buffer holds `room` >= `len` bytes.
    unsafe { data.copy_from_nonoverlapping(value.as_ptr(), value.len()) };
    S_OK
}

/// Null data removes the entry.
extern "system" fn set_private_data(
    this: This,
    id: *const Iid,
    size: u32,
    data: *const u8,
) -> HRESULT {
    // SAFETY: the title passes an id (null is tolerated).
    let Some(&id) = (unsafe { id.as_ref() }) else {
        return E_INVALIDARG;
    };
    let mut map = private_data();
    if data.is_null() {
        map.remove(&(this as usize, id));
    } else {
        // SAFETY: the title passes `size` bytes.
        let value = unsafe { std::slice::from_raw_parts(data, size as usize) };
        map.insert((this as usize, id), value.to_vec());
    }
    S_OK
}

/// Fills a D3D12 table's object methods: private data and name (setting
/// an interface as private data stays unwritten).
pub(crate) const fn d3d12_object<const N: usize>(mut f: [*const (); N]) -> [*const (); N] {
    f[3] = get_private_data as *const (); // 0x18
    f[4] = set_private_data as *const (); // 0x20
    f[6] = set_name as *const (); // 0x30
    f
}

/// The object methods' name setter: names only reach our log.
pub(crate) extern "system" fn set_name(this: This, name: *const u16) -> HRESULT {
    let name = if name.is_null() {
        String::new()
    } else {
        // SAFETY: a non-null name is NUL-terminated UTF-16.
        let len = (0..).take_while(|&i| unsafe { *name.add(i) } != 0).count();
        // SAFETY: `len` units were just read.
        String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(name, len) })
    };
    tracing::debug!(object = ?this, %name, "named");
    S_OK
}

/// The device-child method: the device that made the object, as any of
/// the interfaces the device itself answers to.
pub(crate) extern "system" fn parent_device(_: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let device = crate::device::device();
    answer(
        DEVICE,
        iid,
        crate::device::is_device(iid).then_some(device),
        out,
    )
}

pub(crate) const DEVICE: usize = 0;
pub(crate) const DXGI_DEVICE: usize = 1;
pub(crate) const ADAPTER: usize = 2;
pub(crate) const FACTORY: usize = 3;
pub(crate) const OUTPUT: usize = 4;
pub(crate) const HEAP: usize = 5;
pub(crate) const RESOURCE: usize = 6;
pub(crate) const FENCE: usize = 7;
pub(crate) const QUEUE: usize = 8;
pub(crate) const ALLOCATOR: usize = 9;
pub(crate) const LIST: usize = 10;
pub(crate) const DESCRIPTOR_HEAP: usize = 11;
pub(crate) const ROOT_SIGNATURE: usize = 12;
pub(crate) const QUERY_HEAP: usize = 13;
pub(crate) const COMMAND_SIGNATURE: usize = 14;
pub(crate) const PIPELINE: usize = 15;
const KINDS: [&str; 16] = [
    "device",
    "DXGI device",
    "adapter",
    "factory",
    "output",
    "heap",
    "resource",
    "fence",
    "command queue",
    "command allocator",
    "command list",
    "descriptor heap",
    "root signature",
    "query heap",
    "command signature",
    "pipeline state",
];

const MAX_SLOTS: usize = 160;

/// Which unwritten slots have been reported; each is logged once.
static REPORTED: [[AtomicBool; MAX_SLOTS]; KINDS.len()] =
    [const { [const { AtomicBool::new(false) }; MAX_SLOTS] }; KINDS.len()];

/// Logged once per slot, with the caller's offset in its module (the title
/// image for title calls). Wrong for a slot that returns a value instead of
/// an `HRESULT`.
extern "system" fn missing<const KIND: usize, const SLOT: usize>() -> HRESULT {
    if !REPORTED[KIND][SLOT].swap(true, Ordering::Relaxed) {
        tracing::warn!(
            object = KINDS[KIND],
            slot = format_args!("{:#x}", SLOT * 8),
            caller = caller(),
            "method not implemented"
        );
    }
    E_NOTIMPL
}

/// The return address of whoever called our caller, as `+offset` into its
/// module.
#[inline(never)]
fn caller() -> String {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
        use windows_sys::Win32::System::LibraryLoader::{
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            GetModuleHandleExW,
        };
        let mut frame = std::ptr::null_mut();
        // SAFETY: room for one frame; skips this function and its caller.
        if unsafe { RtlCaptureStackBackTrace(2, 1, &mut frame, std::ptr::null_mut()) } == 1 {
            let mut module = std::ptr::null_mut();
            // SAFETY: looks up the module holding the address, no reference.
            let found = unsafe {
                GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                        | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    frame as *const u16,
                    &mut module,
                )
            };
            if found != 0 {
                return format!("+{:#x}", frame as usize - module as usize);
            }
            return format!("{frame:?}");
        }
    }
    String::new()
}

/// Accepts and ignores (debug-only settings).
pub(crate) extern "system" fn ignore() -> HRESULT {
    S_OK
}

macro_rules! missing {
    ($kind:ident; $($slot:literal)*) => { [$(missing::<$kind, $slot> as *const ()),*] };
}

/// A table of `N` slots that log and fail; callers fill in what they have.
/// Tables are made longer than the interfaces we know, so that a method a
/// newer title adds logs instead of jumping into whatever follows the table.
pub(crate) const fn missing_table<const KIND: usize, const N: usize>() -> [*const (); N] {
    let all = missing!(KIND;
        0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
        32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60
        61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89
        90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113
        114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135
        136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157
        158 159
    );
    assert!(N <= all.len());
    let mut table = [std::ptr::null(); N];
    let mut i = 0;
    while i < N {
        table[i] = all[i];
        i += 1;
    }
    table[1] = add_ref as *const ();
    table[2] = release as *const ();
    table
}

#[cfg(test)]
pub(crate) fn slot<F: Copy>(object: This, offset: usize) -> F {
    // SAFETY: `object` is ours and its table has the slot.
    let f = unsafe { *(*object).functions.add(offset / 8) };
    // SAFETY: the caller names the slot's signature.
    unsafe { std::mem::transmute_copy(&f) }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::mem::offset_of;
    use std::rc::Rc;
    use std::sync::atomic::Ordering;

    use super::*;

    const TEST_IID: Iid = guid(
        0x0102_0304,
        0x0506,
        0x0708,
        [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
    );
    const OTHER_IID: Iid = guid(
        0x090a_0b0c,
        0x0d0e,
        0x0f10,
        [0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27],
    );

    /// A boxed object that counts its own destruction, so that tests can
    /// tell a dropped object from a leaked one.
    #[repr(C)]
    struct Tracked {
        object: Object,
        dropped: Rc<Cell<u32>>,
    }

    impl Tracked {
        /// The header pointer a COM slot receives.
        fn header(&self) -> This {
            &raw const self.object
        }
    }

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.dropped.set(self.dropped.get() + 1);
        }
    }

    /// Only the header and the release slot matter to these tests.
    static TEST_TABLE: Table<8> = {
        let mut f = missing_table::<DEVICE, 8>();
        f[2] = release_boxed::<Tracked> as *const ();
        Table(f)
    };

    static UNREFERENCED: Object = Object::new(&TEST_TABLE);
    static SATURATED: Object = Object::new(&TEST_TABLE);

    fn tracked() -> (Box<Tracked>, Rc<Cell<u32>>) {
        let dropped = Rc::new(Cell::new(0));
        let object = Box::new(Tracked {
            object: Object::new(&TEST_TABLE),
            dropped: Rc::clone(&dropped),
        });
        (object, dropped)
    }

    fn refs(this: This) -> u32 {
        // SAFETY: `this` is one of our test objects.
        unsafe { (*this).refs.load(Ordering::Relaxed) }
    }

    #[test]
    fn guids_round_trip_through_text() {
        let g = guid(
            0x189819f1,
            0x1db6,
            0x4b57,
            [0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7],
        );
        assert_eq!(g[..4], [0xf1, 0x19, 0x98, 0x18]);
        assert_eq!(guid_text(&g), "189819f1-1db6-4b57-be54-1821339b85f7");
        assert_eq!(guid_text(&[0; 16]), "00000000-0000-0000-0000-000000000000");
    }

    #[test]
    fn is_matches_whole_interface_ids_only() {
        assert!(is(&TEST_IID, &[OTHER_IID, TEST_IID]));
        assert!(!is(&OTHER_IID, &[TEST_IID]));
        assert!(!is(std::ptr::null(), &[TEST_IID]));
        assert!(!is(&TEST_IID, &[]));
        let mut altered = TEST_IID;
        altered[15] ^= 1;
        assert!(!is(&altered, &[TEST_IID]));
    }

    #[test]
    fn the_object_header_starts_with_table_bits_and_count() {
        assert_eq!(offset_of!(Object, functions), 0);
        assert_eq!(offset_of!(Object, bits), 8);
        assert_eq!(offset_of!(Object, refs), 12);
        assert_eq!(std::mem::size_of::<Object>(), 16);
    }

    #[test]
    fn answer_reports_no_interface_and_clears_the_result() {
        let (object, dropped) = tracked();
        let header = object.header();
        let mut out = header;
        assert_eq!(answer(DEVICE, &TEST_IID, None, &mut out), E_NOINTERFACE);
        assert!(out.is_null(), "the result is cleared on failure");
        assert_eq!(refs(header), 0, "no reference was added");
        // A null id is tolerated (the log just has no text for it).
        assert_eq!(
            answer(DEVICE, std::ptr::null(), None, &mut out),
            E_NOINTERFACE
        );
        drop(object);
        assert_eq!(dropped.get(), 1);
    }

    #[test]
    fn answer_demands_room_for_the_result() {
        let (object, _dropped) = tracked();
        let header = object.header();
        assert_eq!(
            answer(DEVICE, &TEST_IID, Some(header), std::ptr::null_mut()),
            E_POINTER
        );
        assert_eq!(refs(header), 0, "no reference for a dropped result");
    }

    #[test]
    fn answer_hands_out_a_new_reference() {
        let (object, dropped) = tracked();
        let header = object.header();
        let this: This = Box::into_raw(object).cast::<Object>();
        assert_eq!(header, this);
        let mut out = std::ptr::null();
        assert_eq!(answer(DEVICE, &TEST_IID, Some(this), &mut out), S_OK);
        assert_eq!(out, this);
        assert_eq!(refs(out), 1);
        assert_eq!(release_boxed::<Tracked>(out), 0);
        assert_eq!(dropped.get(), 1);
    }

    #[test]
    fn hand_out_rejects_unknown_interfaces_and_frees_the_object() {
        let (object, dropped) = tracked();
        let mut out: This = std::ptr::null();
        assert_eq!(
            hand_out(DEVICE, object, &OTHER_IID, &[TEST_IID], &mut out),
            E_NOINTERFACE
        );
        assert!(out.is_null());
        assert_eq!(dropped.get(), 1, "the rejected object is dropped");

        let (object, dropped) = tracked();
        let mut out: This = std::ptr::null();
        assert_eq!(
            hand_out(DEVICE, object, std::ptr::null(), &[TEST_IID], &mut out),
            E_NOINTERFACE
        );
        assert!(out.is_null());
        assert_eq!(dropped.get(), 1);
    }

    #[test]
    fn hand_out_without_a_result_only_asks() {
        let (object, dropped) = tracked();
        assert_eq!(
            hand_out(DEVICE, object, &TEST_IID, &[TEST_IID], std::ptr::null_mut()),
            S_FALSE
        );
        assert_eq!(dropped.get(), 1, "the dropped object is freed");
    }

    #[test]
    fn the_table_hands_out_one_reference_and_releases_it() {
        let (object, dropped) = tracked();
        let header = object.header();
        let mut out = std::ptr::null();
        assert_eq!(
            hand_out(DEVICE, object, &TEST_IID, &[TEST_IID], &mut out),
            S_OK
        );
        assert_eq!(out, header);
        assert_eq!(refs(out), 1, "a handed-out object starts with one");

        let add: extern "system" fn(This) -> u32 = slot(out, 0x08);
        let release: extern "system" fn(This) -> u32 = slot(out, 0x10);
        assert_eq!(add(out), 2);
        assert_eq!(release(out), 1);
        assert_eq!(release(out), 0, "the last release frees it");
        assert_eq!(dropped.get(), 1);
    }

    #[test]
    fn release_never_takes_a_count_below_zero() {
        let this: This = &UNREFERENCED;
        assert_eq!(refs(this), 0);
        assert_eq!(release(this), 0, "an unreferenced object stays at zero");
        assert_eq!(refs(this), 0);
        assert_eq!(add_ref(this), 1);
        assert_eq!(release(this), 0);
        assert_eq!(refs(this), 0);
    }

    #[test]
    fn add_ref_saturates_instead_of_wrapping() {
        let this: This = &SATURATED;
        // SAFETY: a static of ours; the count is atomic.
        unsafe { (*this).refs.store(u32::MAX, Ordering::Relaxed) };
        assert_eq!(add_ref(this), u32::MAX, "AddRef never returns 0");
        assert_eq!(refs(this), u32::MAX);
        assert_eq!(release(this), u32::MAX - 1);
    }

    #[test]
    fn tables_begin_with_iunknown_and_fill_unknown_slots() {
        let f = missing_table::<DEVICE, 24>();
        assert_eq!(f[0], missing::<DEVICE, 0> as *const ());
        assert_eq!(f[1], add_ref as *const ());
        assert_eq!(f[2], release as *const ());
        assert_eq!(f[23], missing::<DEVICE, 23> as *const ());
    }

    #[test]
    fn d3d12_object_wires_its_methods_at_the_pc_offsets() {
        let f = d3d12_object(missing_table::<DEVICE, 8>());
        assert_eq!(f[3], get_private_data as *const (), "0x18");
        assert_eq!(f[4], set_private_data as *const (), "0x20");
        assert_eq!(f[5], missing::<DEVICE, 5> as *const (), "0x28");
        assert_eq!(f[6], set_name as *const (), "0x30");
    }

    #[test]
    fn private_data_round_trips_and_reports_a_short_buffer() {
        let (object, dropped) = tracked();
        let mut out = std::ptr::null();
        assert_eq!(
            hand_out(DEVICE, object, &TEST_IID, &[TEST_IID], &mut out),
            S_OK
        );
        let id = guid(0xabc, 1, 2, [3; 8]);
        let value = [0xdeu8, 0xad, 0xbe, 0xef];
        assert_eq!(
            set_private_data(out, &id, value.len() as u32, value.as_ptr()),
            S_OK
        );

        // A null buffer only asks for the size.
        let mut size = 0;
        assert_eq!(
            get_private_data(out, &id, &mut size, std::ptr::null_mut()),
            S_OK
        );
        assert_eq!(size, value.len() as u32);

        // An exact buffer is filled.
        let mut buffer = [0; 4];
        let mut size = buffer.len() as u32;
        assert_eq!(
            get_private_data(out, &id, &mut size, buffer.as_mut_ptr()),
            S_OK
        );
        assert_eq!(buffer, value);
        assert_eq!(size, value.len() as u32);

        // A short buffer is reported along with the size it needs.
        let mut short = [0; 2];
        let mut size = short.len() as u32;
        assert_eq!(
            get_private_data(out, &id, &mut size, short.as_mut_ptr()),
            MORE_DATA
        );
        assert_eq!(size, value.len() as u32);
        assert_eq!(short, [0, 0], "nothing is written on a short buffer");

        // An unknown id is not found.
        let missing_id = guid(0xdef, 4, 5, [6; 8]);
        let mut size = 7;
        assert_eq!(
            get_private_data(out, &missing_id, &mut size, std::ptr::null_mut()),
            NOT_FOUND
        );
        assert_eq!(size, 0);

        // Null data removes the entry.
        assert_eq!(set_private_data(out, &id, 0, std::ptr::null()), S_OK);
        let mut size = 0;
        assert_eq!(
            get_private_data(out, &id, &mut size, std::ptr::null_mut()),
            NOT_FOUND
        );

        assert_eq!(release_boxed::<Tracked>(out), 0);
        assert_eq!(dropped.get(), 1);
    }

    #[test]
    fn private_data_requires_its_arguments() {
        let (object, dropped) = tracked();
        let mut out = std::ptr::null();
        assert_eq!(
            hand_out(DEVICE, object, &TEST_IID, &[TEST_IID], &mut out),
            S_OK
        );
        let id = guid(0x123, 7, 8, [9; 8]);
        let mut size = 0;
        assert_eq!(
            get_private_data(out, std::ptr::null(), &mut size, std::ptr::null_mut()),
            E_INVALIDARG
        );
        assert_eq!(
            get_private_data(out, &id, std::ptr::null_mut(), std::ptr::null_mut()),
            E_INVALIDARG
        );
        assert_eq!(
            set_private_data(out, std::ptr::null(), 0, std::ptr::null()),
            E_INVALIDARG
        );
        assert_eq!(release_boxed::<Tracked>(out), 0);
        assert_eq!(dropped.get(), 1);
    }

    #[test]
    fn private_data_goes_away_with_its_object() {
        let (object, dropped) = tracked();
        let mut out = std::ptr::null();
        assert_eq!(
            hand_out(DEVICE, object, &TEST_IID, &[TEST_IID], &mut out),
            S_OK
        );
        let id = guid(0x456, 10, 11, [12; 8]);
        let value = [7u8; 3];
        assert_eq!(
            set_private_data(out, &id, value.len() as u32, value.as_ptr()),
            S_OK
        );
        let key = (out as usize, id);
        assert!(private_data().contains_key(&key));
        assert_eq!(release_boxed::<Tracked>(out), 0);
        assert!(!private_data().contains_key(&key), "the data is dropped");
        assert_eq!(dropped.get(), 1);
    }

    #[test]
    fn set_name_accepts_absent_and_present_names() {
        assert_eq!(set_name(std::ptr::null(), std::ptr::null()), S_OK);
        let empty = [0u16];
        assert_eq!(set_name(std::ptr::null(), empty.as_ptr()), S_OK);
        let name = [u16::from(b'x'), 0];
        assert_eq!(set_name(std::ptr::null(), name.as_ptr()), S_OK);
        // An unpaired surrogate is replaced, not a panic.
        let broken = [0xd800u16, 0];
        assert_eq!(set_name(std::ptr::null(), broken.as_ptr()), S_OK);
    }

    #[test]
    fn parent_device_answers_only_the_device_ids() {
        let mut out = std::ptr::null();
        assert_eq!(
            parent_device(std::ptr::null(), &crate::device::IID, &mut out),
            S_OK
        );
        assert_eq!(out, crate::device::device());
        release(out);

        let mut out = crate::device::device();
        assert_eq!(
            parent_device(std::ptr::null(), &OTHER_IID, &mut out),
            E_NOINTERFACE
        );
        assert!(out.is_null());

        assert_eq!(
            parent_device(std::ptr::null(), &crate::device::IID, std::ptr::null_mut()),
            E_POINTER
        );

        let mut out = crate::device::device();
        assert_eq!(
            parent_device(std::ptr::null(), std::ptr::null(), &mut out),
            E_NOINTERFACE
        );
        assert!(out.is_null());
    }

    /// Every COM object answers for `IUnknown`, and with one pointer
    /// whichever of its interfaces is asked: the object's identity.
    #[test]
    fn every_object_answers_for_iunknown() {
        type Query = extern "system" fn(This, *const Iid, *mut This) -> HRESULT;
        let unknown = |object: This| {
            let mut found = std::ptr::null();
            let hr = slot::<Query>(object, 0)(object, &IUNKNOWN, &mut found);
            (hr, found)
        };
        let device = crate::device::device();
        let mut fence = std::ptr::null();
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(std::ptr::null(), 0, 0, fence_iid, &mut fence),
            S_OK
        );
        assert_eq!(unknown(device), (S_OK, device));
        assert_eq!(unknown(fence), (S_OK, fence));
        // The DXGI device is the same object as the device it answers for.
        assert_eq!(unknown(crate::dxgi::device()), (S_OK, device));
        // An id nobody has is still nobody's.
        let mut found = fence;
        let nobodys = guid(0x189819f1, 0x1db6, 0x4b57, [0xbe; 8]);
        assert_eq!(
            slot::<Query>(fence, 0)(fence, &nobodys, &mut found),
            E_NOINTERFACE
        );
        assert!(found.is_null());
        slot::<extern "system" fn(This) -> u32>(fence, 0x10)(fence);
    }
}
