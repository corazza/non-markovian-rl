pub use crate::learner::{TabularLearner, TabularLearnerConfig, TabularLearnerData};
use crate::mdp::MDP;

pub struct QLearning<E: MDP> {
    pub config: TabularLearnerConfig,
    data: TabularLearnerData<E>,
}

impl<E: MDP> QLearning<E> {
    pub fn new(config: TabularLearnerConfig, terminal_state: E::State) -> QLearning<E> {
        let data = TabularLearnerData::new(terminal_state);
        QLearning { config, data }
    }
}

impl<E: MDP> TabularLearner<E> for QLearning<E> {
    // env is preinitialized
    fn episode(&mut self, env: &mut E) {
        self.data.terminal_state = env.get_terminal();
        let mut state = env.current_state();

        loop {
            let action = self.epsilon_greedy(self.config.epsilon, env.current_state(), env);
            let (next_state, reward) = env.take_action(action).unwrap();
            let target = reward + self.config.gamma * self.max_action_value(next_state, &env);
            self.update(self.config.alpha, state, action, target);

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
