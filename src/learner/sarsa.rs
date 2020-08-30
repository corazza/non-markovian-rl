use std::collections::HashMap;

pub use crate::learner::TabularLearner;
use crate::mdp::{Reward, MDP};

pub struct Sarsa<E: MDP> {
    alpha: Reward, // learning rate
    epsilon: f32,  // epsilon-greedy
    gamma: f32,    // discount factor
    q: HashMap<(E::State, E::Action), Reward>,
    initial_q: Reward,
    debug: bool,
    terminal_state: E::State,
}

impl<E: MDP> Sarsa<E> {
    pub fn new(
        alpha: Reward,
        epsilon: f32,
        gamma: f32,
        initial_q: Reward,
        terminal_state: E::State,
    ) -> Sarsa<E> {
        Sarsa {
            alpha,
            epsilon,
            gamma,
            q: HashMap::new(),
            initial_q,
            debug: false,
            terminal_state,
        }
    }
}

impl<E: MDP> TabularLearner<E> for Sarsa<E> {
    fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    fn set_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon;
    }

    fn alpha(&self) -> f32 {
        self.alpha
    }

    fn value(&self, state: E::State, action: E::Action) -> Reward {
        if state == self.terminal_state {
            0.
        } else {
            *self.q.get(&(state, action)).unwrap_or(&self.initial_q)
        }
    }

    fn set_value(&mut self, state: E::State, action: E::Action, value: Reward) {
        self.q.insert((state, action), value);
    }

    // env is preinitialized
    fn episode(&mut self, env: &mut E) {
        self.terminal_state = env.get_terminal();
        let mut action = self.epsilon_greedy(self.epsilon, env.current_state(), env);
        let mut state = env.current_state();

        while let Some((next_state, reward)) = env.take_action(action) {
            let next_action = self.epsilon_greedy(self.epsilon, next_state, env);
            if self.debug {
                println!(
                    "S: {:?}, A: {:?}, R: {}, S': {:?}, A': {:?}",
                    state, action, reward, next_state, next_action
                );
            }
            let target = reward + self.gamma * self.value(next_state, next_action);
            self.update(state, action, target);
            state = next_state;
            action = next_action;
        }
    }
}
