//! Live diagnostic gamepad input, adapted from ScarlettPort's VICTORY_PRESS_FILE
//! (runtime/victory, commit 48ba23a, GPL-3.0). Write a whitespace-separated
//! sequence such as "DOWN ENTER" to the configured file. Each pulse includes a
//! release long enough for menus polling the current state at three fps.

use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Read;
use std::os::windows::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::{A, B, DOWN, LEFT, MENU, RIGHT, UP, VIEW, X, Y};

const PULSE: Duration = Duration::from_millis(420);
const POLL: Duration = Duration::from_millis(200);

pub(super) struct PressFile {
    path: PathBuf,
    pending: VecDeque<u32>,
    held: Option<(u32, Instant)>,
    next: Option<Instant>,
}

impl PressFile {
    pub(super) fn new(path: PathBuf) -> Self {
        Self {
            path,
            pending: VecDeque::new(),
            held: None,
            next: None,
        }
    }

    pub(super) fn sample(&mut self, now: Instant) -> Option<u32> {
        if let Some((button, until)) = self.held {
            if now < until {
                return Some(button);
            }
            self.held = None;
            self.next = Some(now + PULSE);
        }
        if self.next.is_some_and(|next| now < next) {
            return None;
        }
        if self.pending.is_empty() {
            self.next = Some(now + POLL);
            // Hold the file exclusively while reading and acknowledging this batch.
            // A writer replacing it cannot lose a newer command to our truncation.
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .share_mode(0)
                .open(&self.path)
                .ok()?;
            let mut text = String::new();
            file.read_to_string(&mut text).ok()?;
            if text.is_empty() {
                return None;
            }
            file.set_len(0).ok()?;
            self.pending.extend(
                text.trim_start_matches('\u{feff}')
                    .split_whitespace()
                    .filter_map(button),
            );
        }
        let button = self.pending.pop_front()?;
        self.held = Some((button, now + PULSE));
        Some(button)
    }
}

fn button(token: &str) -> Option<u32> {
    Some(match token.to_ascii_uppercase().as_str() {
        "A" | "ENTER" => A,
        "B" | "BACK" => B,
        "X" => X,
        "Y" => Y,
        "UP" => UP,
        "DOWN" => DOWN,
        "LEFT" => LEFT,
        "RIGHT" => RIGHT,
        "START" | "MENU" | "ESC" => MENU,
        "VIEW" | "TAB" => VIEW,
        _ => return None,
    })
}

#[test]
fn delayed_polls_release_each_button_before_the_next_press() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("press.txt");
    let mut input = PressFile::new(path.clone());
    let start = Instant::now();
    let at = |ms| start + Duration::from_millis(ms);
    std::fs::write(&path, "\u{feff}down DOWN Enter").unwrap();
    assert_eq!(input.sample(at(0)), Some(DOWN));
    assert_eq!(std::fs::read(&path).unwrap(), b"");
    std::fs::write(&path, "A").unwrap();
    assert_eq!(input.sample(at(334)), Some(DOWN));
    assert_eq!(input.sample(at(667)), None);
    assert_eq!(input.sample(at(1000)), None);
    assert_eq!(input.sample(at(1334)), Some(DOWN));
    assert_eq!(input.sample(at(2000)), None);
    assert_eq!(input.sample(at(2500)), Some(A));
    assert_eq!(input.sample(at(3000)), None);
    assert_eq!(input.sample(at(3500)), Some(A));
    assert_eq!(std::fs::read(&path).unwrap(), b"");
}
