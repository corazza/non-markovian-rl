use super::{Environment, Reward};
use rand::Rng;

// State representation exposed to the agent
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum TCorridorState {
    Start,
    ObserveU, // Marks upper state as trapped
    ObserveL, // Marks lower state as trapped
    Split,    // Trap location no longer observable
    Terminal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TCorridorAction {
    Forward,
    Backward,
    Up,
    Down,
}

pub struct TCorridor {
    current_state: TCorridorState,
    observed: i8, // -1 is lower, 1 is upper
}

impl TCorridor {
    pub fn new() -> TCorridor {
        TCorridor {
            current_state: TCorridorState::Start,
            observed: 0,
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
}

impl Environment for TCorridor {
    type Action = TCorridorAction;
    type State = TCorridorState;

    fn take_action(&mut self, action: Self::Action) -> Option<(Self::State, Reward)> {
        if self.terminated() {
            return None;
        }

        use TCorridorAction::*;
        use TCorridorState::*;
        let (next_state, reward) = match (self.current_state, action) {
            (Start, TCorridorAction::Forward) => (self.observe(), 0.0),
            (Start, _) => (Start, 0.0),
            (ObserveL, Forward) => (Split, 0.0),
            (ObserveL, Backward) => (Start, 0.0),
            (ObserveL, _) => (ObserveL, 0.0),
            (ObserveU, Forward) => (Split, 0.0),
            (ObserveU, Backward) => (Start, 0.0),
            (ObserveU, _) => (ObserveU, 0.0),
            (Split, Up) => (Terminal, if self.observed == 1 { -1.0 } else { 0.0 }),
            (Split, Down) => (Terminal, if self.observed == -1 { -1.0 } else { 0.0 }),
            (Split, Backward) => (
                if self.observed == 1 {
                    ObserveU
                } else {
                    ObserveL
                },
                0.0,
            ),
            (Split, _) => (Split, 0.0),
            (Terminal, _) => (Terminal, 0.0), // can't happen

                                              // (Start, _) => (self.observe(), 0.0),
                                              // (ObserveL, _) => (Split, 0.0),
                                              // (ObserveU, _) => (Split, 0.0),
                                              // (Split, Up) => (Terminal, if self.observed == 1 { -1.0 } else { 0.0 }),
                                              // (Split, Down) => (Terminal, if self.observed == -1 { -1.0 } else { 0.0 }),
                                              // (Split, _) => (Split, 0.0),
                                              // (Terminal, _) => (Terminal, 0.0), // can't happen
        };

        self.current_state = next_state;
        Some((next_state, reward))
    }

    fn available_actions(&self, state: Self::State) -> Vec<Self::Action> {
        use TCorridorAction::*;
        use TCorridorState::*;

        match state {
            Start => vec![Forward],
            ObserveL => vec![Forward],
            ObserveU => vec![Forward],
            Split => vec![Up, Down],
            Terminal => vec![Forward], // implementational detail
        }

        // vec![Forward, Backward, Up, Down]
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
