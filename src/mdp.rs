use std::hash::Hash;

pub type Reward = f32;

/// Interface for a Markov decision process
pub trait MDP {
    type Action: Copy + Hash + Eq + std::fmt::Debug;
    // an index
    type State: Copy + Hash + Eq + std::fmt::Debug;

    /// Returns None on terminal state
    fn take_action(&mut self, action: Self::Action) -> Option<(Self::State, Reward)>;

    /// Returns all actions available at state
    fn available_actions(&self, state: Self::State) -> Vec<Self::Action>;

    fn current_state(&self) -> Self::State;
}
