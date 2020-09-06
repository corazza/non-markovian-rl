use std::collections::HashMap;

pub use crate::learner::{TabularLearner, TabularLearnerConfig, TabularLearnerData};
use crate::mdp::{Reward, MDP};

pub struct Sarsa<E: MDP> {
    pub config: TabularLearnerConfig,
    data: TabularLearnerData<E>,
}

impl<E: MDP> Sarsa<E> {
    pub fn new(config: TabularLearnerConfig, terminal_state: E::State) -> Sarsa<E> {
        let data = TabularLearnerData::new(terminal_state);
        Sarsa { config, data }
    }
}

impl<E: MDP> TabularLearner<E> for Sarsa<E> {
    // env is preinitialized
    fn episode(&mut self, env: &mut E) {
        self.data.terminal_state = env.get_terminal();
        let mut action = self.epsilon_greedy(self.config.epsilon, env.current_state(), env);
        let mut state = env.current_state();

        while let Some((next_state, reward)) = env.take_action(action) {
            let next_action = self.epsilon_greedy(self.config.epsilon, next_state, env);
            if self.config.debug {
                println!(
                    "S: {:?}, A: {:?}, R: {}, S': {:?}, A': {:?}",
                    state, action, reward, next_state, next_action
                );
            }
            let target =
                reward + self.config.gamma * self.data.value(&self.config, next_state, next_action);
            self.update(self.config.alpha, state, action, target);
            state = next_state;
            action = next_action;
        }
    }

    fn data(&self) -> &TabularLearnerData<E> {
        &self.data
    }

    fn data_mut(&mut self) -> &mut TabularLearnerData<E> {
        &mut self.data
    }

    fn config(&self) -> &TabularLearnerConfig {
        &self.config
    }
}
