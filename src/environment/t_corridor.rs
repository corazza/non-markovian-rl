use super::{Environment, Reward};
use rand::Rng;

// State representation exposed to the agent
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum TCorridorState {
    Start,
    ObserveU,        // Marks upper state as trapped
    ObserveL,        // Marks lower state as trapped
    Corridor(usize), // regular part of the corridor
    Split,           // Trap location no longer observable
    Terminal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TCorridorAction {
    Forward,
    Backward,
    Up,
    Down,
}

pub struct TCorridor {
    current_state: TCorridorState,
    observed: i8, // -1 is lower, 1 is upper
    normal_steps: usize,
}

impl TCorridor {
    pub fn new(normal_steps: usize) -> TCorridor {
        TCorridor {
            current_state: TCorridorState::Start,
            observed: 0,
            normal_steps,
        }
    }

    fn observe(&mut self) -> TCorridorState {
        if self.observed == 1 {
            return TCorridorState::ObserveU;
        }

        if self.observed == -1 {
            return TCorridorState::ObserveL;
        }

        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < 0.5 {
            self.observed = 1;
            TCorridorState::ObserveU
        } else {
            self.observed = -1;
            TCorridorState::ObserveL
        }
    }

    fn split_or_corridor(&self) -> TCorridorState {
        use TCorridorState::*;
        if self.normal_steps > 0 {
            Corridor(1)
        } else {
            Split
        }
    }
}

impl Environment for TCorridor {
    type Action = TCorridorAction;
    type State = TCorridorState;

    fn take_action(&mut self, action: Self::Action) -> Option<(Self::State, Reward)> {
        let default_reward: Reward = -5.0;
        let trap_reward: Reward = -100.0;
        let nontrap_reward: Reward = 100.0;

        if self.terminated() {
            return None;
        }

        use TCorridorAction::*;
        use TCorridorState::*;
        let (next_state, reward) = match (self.current_state, action) {
            (Start, TCorridorAction::Forward) => (self.observe(), default_reward),
            (Start, _) => (Start, default_reward),
            (ObserveL, Forward) => (self.split_or_corridor(), default_reward),
            (ObserveL, Backward) => (Start, default_reward),
            (ObserveL, _) => (ObserveL, default_reward),
            (ObserveU, Forward) => (self.split_or_corridor(), default_reward),
            (ObserveU, Backward) => (Start, default_reward),
            (ObserveU, _) => (ObserveU, default_reward),
            (Corridor(n), Forward) => {
                if n == self.normal_steps {
                    (Split, default_reward)
                } else {
                    (Corridor(n + 1), default_reward)
                }
            }
            (Corridor(n), _) => (Corridor(n), default_reward),
            (Split, Up) => (
                Terminal,
                if self.observed == 1 {
                    trap_reward
                } else {
                    nontrap_reward
                },
            ),
            (Split, Down) => (
                Terminal,
                if self.observed == -1 {
                    trap_reward
                } else {
                    nontrap_reward
                },
            ),
            (Split, Backward) => (
                if self.observed == 1 {
                    ObserveU
                } else {
                    ObserveL
                },
                default_reward,
            ),
            (Split, _) => (Split, default_reward),
            (Terminal, _) => panic!("action attempted in terminal"),
        };

        self.current_state = next_state;
        Some((next_state, reward))
    }

    fn available_actions(&self, state: Self::State) -> Vec<Self::Action> {
        use TCorridorAction::*;
        use TCorridorState::*;

        match state {
            // Start => vec![Forward],
            // ObserveL => vec![Forward, Backward],
            // ObserveU => vec![Forward, Backward],
            // Corridor(_) => vec![Forward, Backward],
            // Split => vec![Up, Down, Backward],
            // Terminal => vec![Forward],
            Start => vec![Forward],
            ObserveL => vec![Forward],
            ObserveU => vec![Forward],
            Corridor(_) => vec![Forward],
            Split => vec![Up, Down],
            Terminal => vec![Forward],
        }
    }

    fn current_state(&self) -> Self::State {
        self.current_state
    }

    fn terminated(&self) -> bool {
        self.current_state == TCorridorState::Terminal
    }

    fn is_terminal(&self, state: Self::State) -> bool {
        state == TCorridorState::Terminal
    }

    fn get_terminal(&self) -> Self::State {
        TCorridorState::Terminal
    }
}
