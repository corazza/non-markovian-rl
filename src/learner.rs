use crate::mdp::{Reward, MDP};
use rand::Rng;

pub use self::q_learning::QLearning;
pub use self::sarsa::Sarsa;
pub mod q_learning;
pub mod sarsa;

pub trait TabularLearner<E: MDP> {
    fn set_epsilon(&mut self, epsilon: f32);
    fn set_debug(&mut self, debug: bool);
    fn alpha(&self) -> f32;

    fn episode(&mut self, env: &mut E);

    fn value(&self, state: E::State, action: E::Action) -> Reward;
    fn set_value(&mut self, state: E::State, action: E::Action, value: Reward);
    fn update(&mut self, state: E::State, action: E::Action, target: Reward) {
        let current_value = self.value(state, action);
        self.set_value(
            state,
            action,
            current_value + self.alpha() * (target - current_value),
        );
    }

    fn epsilon_greedy(&self, epsilon: f32, from: E::State, env: &E) -> E::Action {
        let available = env.available_actions(from);
        let mut with_values: Vec<(E::Action, Reward)> = available
            .iter()
            .map(|action| (*action, self.value(from, *action)))
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
            .map(|action| (*action, self.value(state, *action)))
            .collect();
        with_values.sort_by(|(_, v1), (_, v2)| v2.partial_cmp(v1).unwrap());
        with_values[0].1
    }
}
