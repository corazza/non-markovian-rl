use rand::Rng;
use std::collections::HashMap;

pub use crate::learner::{TabularLearner, TabularLearnerConfig, TabularLearnerData};
use crate::mdp::{Reward, MDP};

pub struct DynaQ<E: MDP> {
    pub config: TabularLearnerConfig,
    data: TabularLearnerData<E>,
    n: u32, // planning steps (when planning is used, e.g. DynaQ)
    model: HashMap<(E::State, E::Action), (E::State, Reward)>,
}

impl<E: MDP> DynaQ<E> {
    pub fn new(config: TabularLearnerConfig, n: u32, terminal_state: E::State) -> DynaQ<E> {
        let data = TabularLearnerData::new(terminal_state);
        DynaQ {
            config,
            data,
            n,
            model: HashMap::new(),
        }
    }
}

impl<E: MDP> TabularLearner<E> for DynaQ<E> {
    // env is preinitialized
    fn episode(&mut self, env: &mut E) {
        self.data.terminal_state = env.get_terminal();

        loop {
            let mut state = env.current_state();
            let action = self.epsilon_greedy(self.config.epsilon, env.current_state(), env);
            let (next_state, reward) = env.take_action(action).unwrap();
            let target = reward + self.config.gamma * self.max_action_value(next_state, &env);
            self.update(self.config.alpha, state, action, target);
            self.model.insert((state, action), (next_state, reward));

            let mut rng = rand::thread_rng();

            for _ in 0..self.n {
                let model_keys: Vec<(E::State, E::Action)> =
                    self.model.keys().map(|a| *a).collect();
                let (model_state, model_action) = model_keys[rng.gen_range(0, model_keys.len())];
                let (model_next_state, model_reward) = self.model[&(model_state, model_action)];
                let target = model_reward
                    + self.config.gamma * self.max_action_value(model_next_state, &env);
                self.update(self.config.alpha, model_state, model_action, target);
            }

            if self.config.debug {
                println!("{:?} -> {:?}", state, next_state);
            }

            state = next_state;
            if state == self.data.terminal_state {
                break;
            }
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
