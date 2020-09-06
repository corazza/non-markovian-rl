use rand::Rng;
use std::collections::HashMap;

pub use crate::learner::TabularLearner;
use crate::mdp::{Reward, MDP};

pub struct DynaQ<E: MDP> {
    alpha: Reward, // learning rate
    epsilon: f32,  // epsilon-greedy
    gamma: f32,    // discount factor
    n: u32,        // planning steps
    q: HashMap<(E::State, E::Action), Reward>,
    model: HashMap<(E::State, E::Action), (E::State, Reward)>,
    initial_q: Reward,
    debug: bool,
    terminal_state: E::State,
}

impl<E: MDP> DynaQ<E> {
    pub fn new(
        alpha: Reward,
        epsilon: f32,
        gamma: f32,
        n: u32,
        initial_q: Reward,
        terminal_state: E::State,
    ) -> DynaQ<E> {
        DynaQ {
            alpha,
            epsilon,
            gamma,
            n,
            q: HashMap::new(),
            model: HashMap::new(),
            initial_q,
            debug: false,
            terminal_state,
        }
    }
}

impl<E: MDP> TabularLearner<E> for DynaQ<E> {
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

        loop {
            let mut state = env.current_state();
            let action = self.epsilon_greedy(self.epsilon, env.current_state(), env);
            let (next_state, reward) = env.take_action(action).unwrap();
            let target = reward + self.gamma * self.max_action_value(next_state, &env);
            self.update(state, action, target);
            self.model.insert((state, action), (next_state, reward));

            let mut rng = rand::thread_rng();

            for _ in 0..self.n {
                let model_keys: Vec<(E::State, E::Action)> =
                    self.model.keys().map(|a| *a).collect();
                let (model_state, model_action) = model_keys[rng.gen_range(0, model_keys.len())];
                let (model_next_state, model_reward) = self.model[&(model_state, model_action)];
                let target =
                    model_reward + self.gamma * self.max_action_value(model_next_state, &env);
                self.update(model_state, model_action, target);
            }

            if self.debug {
                println!("{:?} -> {:?}", state, next_state);
            }

            state = next_state;
            if state == self.terminal_state {
                break;
            }
        }
    }
}
