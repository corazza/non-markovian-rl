use std::collections::VecDeque;

use crate::environment::{Environment, Reward};
pub use crate::learner::{TabularLearner, TabularLearnerConfig, TabularLearnerData};

pub struct NStepSarsa<E: Environment> {
    pub config: TabularLearnerConfig,
    data: TabularLearnerData<E>,
    n: usize,                                         // steps
    history: VecDeque<(E::State, E::Action, Reward)>, // last n triplets (S_t, A_t, R_{t+1})
}

impl<E: Environment> NStepSarsa<E> {
    pub fn new(n: usize, config: TabularLearnerConfig, terminal_state: E::State) -> NStepSarsa<E> {
        let data = TabularLearnerData::new(terminal_state);
        NStepSarsa {
            config,
            data,
            n,
            history: VecDeque::with_capacity(n),
        }
    }
}

impl<E: Environment> TabularLearner<E> for NStepSarsa<E> {
    // env is preinitialized
    fn episode(&mut self, env: &mut E) -> Reward {
        self.data.terminal_state = env.get_terminal();
        let mut action = self.epsilon_greedy(self.config.epsilon, env.current_state(), env);
        let mut state = env.current_state();
        let mut gain: Reward = 0.0;

        while let Some((next_state, reward)) = env.take_action(action) {
            // episode() assumes gamma=1
            gain += reward;
            let next_action = self.epsilon_greedy(self.config.epsilon, next_state, env);
            if self.config.debug {
                println!(
                    "S: {:?}, A: {:?}, R: {}, S': {:?}, A': {:?}",
                    state, action, reward, next_state, next_action
                );
            }

            if self.history.len() == self.n {
                let mut target = self.history[self.n - 1].2;

                for i in (0..self.n - 1).rev() {
                    target = self.history[i].2 + self.config.gamma * target;
                }

                target += self.config.gamma.powf(self.n as f32)
                    * self.data.value(&self.config, state, action);

                self.update(
                    self.config.alpha,
                    self.history[0].0,
                    self.history[0].1,
                    target,
                );

                self.history.pop_front();
            }

            self.history.push_back((state, action, reward));

            state = next_state;
            action = next_action;
        }

        assert!(self.history.len() <= self.n);
        let n = self.history.len(); // to handle cases where n_states < self.n

        for i in 0..n {
            let mut target = self.history[n - 1].2;

            for j in (i..n - 1).rev() {
                target = self.history[j].2 + self.config.gamma * target;
            }

            self.update(
                self.config.alpha,
                self.history[i].0,
                self.history[i].1,
                target,
            );
        }

        self.history.clear();

        gain
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

    fn config_mut(&mut self) -> &mut TabularLearnerConfig {
        &mut self.config
    }
}
