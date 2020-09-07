use std::collections::VecDeque;

pub use crate::learner::{TabularLearner, TabularLearnerConfig, TabularLearnerData};
use crate::mdp::{MDP, Reward};

pub struct NStepSarsa<E: MDP> {
    pub config: TabularLearnerConfig,
    data: TabularLearnerData<E>,
    n: usize, // steps
    history: VecDeque<(E::State, E::Action, Reward)>, // last n triplets (S_t, A_t, R_{t+1})
}

impl<E: MDP> NStepSarsa<E> {
    pub fn new(n: usize, config: TabularLearnerConfig, terminal_state: E::State) -> NStepSarsa<E> {
        let data = TabularLearnerData::new(terminal_state);
        NStepSarsa { config, data, n, history: VecDeque::with_capacity(n) }
    }
}

impl<E: MDP> TabularLearner<E> for NStepSarsa<E> {
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

            if self.history.len() == self.n {
                let mut ret = self.history[0].2;

                for i in 1..self.history.len() {
                    ret = self.config.gamma*(ret + self.history[i].2);
                }

                let target = reward + self.config.gamma * self.data.value(&self.config, next_state, next_action);
                self.update(self.config.alpha, state, action, target);
            }

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
