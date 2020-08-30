use crate::mdp::{Reward, MDP};
use rand::Rng;
use std::collections::HashMap;

pub trait StateActionLearner {}

pub struct Sarsa<E: MDP> {
    alpha: Reward,    // learning rate
    pub epsilon: f32, // epsilon-greedy
    gamma: f32,       // discount factor
    q: HashMap<(E::State, E::Action), Reward>,
    initial_q: Reward,
    pub debug: bool,
}

impl<E: MDP> Sarsa<E> {
    pub fn new(alpha: Reward, epsilon: f32, gamma: f32, initial_q: Reward) -> Sarsa<E> {
        Sarsa {
            alpha,
            epsilon,
            gamma,
            q: HashMap::new(),
            initial_q,
            debug: false,
        }
    }

    pub fn value(&self, state: E::State, action: E::Action) -> Reward {
        *self.q.get(&(state, action)).unwrap_or(&self.initial_q)
    }

    pub fn epsilon_greedy(&self, from: E::State, env: &E) -> E::Action {
        let available = env.available_actions(from);
        let mut with_values: Vec<(E::Action, Reward)> = available
            .iter()
            .map(|action| (*action, self.value(from, *action)))
            .collect();
        with_values.sort_by(|(_, v1), (_, v2)| v2.partial_cmp(v1).unwrap());

        let mut rng = rand::thread_rng();

        if rng.gen::<f32>() < self.epsilon {
            with_values[rng.gen_range(0, with_values.len())].0
        } else {
            with_values[0].0
        }
    }

    pub fn update(&mut self, state: E::State, action: E::Action, target: Reward) {
        let current_value = self.value(state, action);
        self.q.insert(
            (state, action),
            current_value + self.alpha * (target - current_value),
        );
    }

    // env is preinitialized
    pub fn episode(&mut self, env: &mut E) {
        let mut action = self.epsilon_greedy(env.current_state(), env);
        let mut state = env.current_state();

        while let Some((next_state, reward)) = env.take_action(action) {
            let next_action = self.epsilon_greedy(next_state, env);
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
