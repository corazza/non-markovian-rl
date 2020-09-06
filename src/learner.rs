use crate::mdp::{Reward, MDP};
use rand::Rng;
use std::collections::HashMap;

pub use self::dyna_q::DynaQ;
pub use self::q_learning::QLearning;
pub use self::sarsa::Sarsa;
pub mod dyna_q;
pub mod q_learning;
pub mod sarsa;

pub struct TabularLearnerData<E: MDP> {
    pub q: HashMap<(E::State, E::Action), Reward>,
    pub terminal_state: E::State,
}

impl<E: MDP> TabularLearnerData<E> {
    pub fn new(terminal_state: E::State) -> TabularLearnerData<E> {
        TabularLearnerData {
            q: HashMap::new(),
            terminal_state,
        }
    }

    fn value(&self, config: &TabularLearnerConfig, state: E::State, action: E::Action) -> Reward {
        if state == self.terminal_state {
            0.
        } else {
            *self.q.get(&(state, action)).unwrap_or(&config.initial_q)
        }
    }

    fn set_value(&mut self, state: E::State, action: E::Action, value: Reward) {
        self.q.insert((state, action), value);
    }
}

#[derive(Clone)]
pub struct TabularLearnerConfig {
    alpha: f32,        // learning rate
    pub epsilon: f32,  // epsilon-greedy
    gamma: f32,        // discount factor
    pub debug: bool,   // print episode steps
    initial_q: Reward, // default value
}

impl TabularLearnerConfig {
    pub fn new(alpha: f32, epsilon: f32, gamma: f32, initial_q: Reward) -> TabularLearnerConfig {
        TabularLearnerConfig {
            alpha,
            epsilon,
            gamma,
            debug: false,
            initial_q,
        }
    }
}

pub trait TabularLearner<E: MDP> {
    fn episode(&mut self, env: &mut E);
    fn data(&self) -> &TabularLearnerData<E>;
    fn data_mut(&mut self) -> &mut TabularLearnerData<E>;
    fn config(&self) -> &TabularLearnerConfig;
    fn update(&mut self, alpha: f32, state: E::State, action: E::Action, target: Reward) {
        let current_value = self.data().value(self.config(), state, action);
        self.data_mut().set_value(
            state,
            action,
            current_value + alpha * (target - current_value),
        );
    }

    fn epsilon_greedy(&self, epsilon: f32, from: E::State, env: &E) -> E::Action {
        let available = env.available_actions(from);
        let mut with_values: Vec<(E::Action, Reward)> = available
            .iter()
            .map(|action| (*action, self.data().value(self.config(), from, *action)))
            .collect();
        with_values.sort_by(|(_, v1), (_, v2)| v2.partial_cmp(v1).unwrap());

        let mut rng = rand::thread_rng();

        if rng.gen::<f32>() < epsilon {
            with_values[rng.gen_range(0, with_values.len())].0
        } else {
            with_values[0].0
        }
    }

    fn max_action_value(&self, state: E::State, env: &E) -> f32 {
        let available = env.available_actions(state);
        let mut with_values: Vec<(E::Action, Reward)> = available
            .iter()
            .map(|action| (*action, self.data().value(self.config(), state, *action)))
            .collect();
        with_values.sort_by(|(_, v1), (_, v2)| v2.partial_cmp(v1).unwrap());
        with_values[0].1
    }
}
