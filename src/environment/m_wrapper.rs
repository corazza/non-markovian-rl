use super::{Environment, Reward};

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum MAction {
    Flip,
    Noop,
}

impl MAction {
    fn bit_value(self, m: u8) -> u8 {
        match self {
            MAction::Flip => {
                if m == 1 {
                    0
                } else {
                    1
                }
            }
            MAction::Noop => m,
        }
    }
}

pub struct MWrapper<E: Environment> {
    env: E, // environment to wrap
    m: u8,  // memory bits
}

impl<E: Environment> MWrapper<E> {
    pub fn new(env: E) -> MWrapper<E> {
        MWrapper { env, m: 0 }
    }
}

impl<E: Environment> Environment for MWrapper<E> {
    type Action = (E::Action, MAction);
    type State = (E::State, u8);

    fn take_action(
        &mut self,
        (env_action, m_action): Self::Action,
    ) -> Option<(Self::State, Reward)> {
        match self.env.take_action(env_action) {
            None => None,
            Some((next_env_state, reward)) => {
                self.m = m_action.bit_value(self.m);
                if self.env.terminated() {
                    self.m = 2;
                }
                Some(((next_env_state, self.m), reward))
            }
        }
    }

    fn available_actions(&self, (env_state, _m_state): Self::State) -> Vec<Self::Action> {
        self.env
            .available_actions(env_state)
            .iter()
            .map(|env_action| vec![(*env_action, MAction::Flip), (*env_action, MAction::Noop)])
            .flatten()
            .collect()
    }

    fn current_state(&self) -> Self::State {
        (self.env.current_state(), self.m)
    }

    fn terminated(&self) -> bool {
        if self.env.terminated() || self.m == 2 {
            assert!(self.env.terminated() && self.m == 2);
            return true;
        } else {
            return false;
        }
    }

    fn is_terminal(&self, (env_state, m_state): Self::State) -> bool {
        if self.env.is_terminal(env_state) || m_state == 2 {
            assert!(self.env.is_terminal(env_state) && m_state == 2);
            return true;
        } else {
            return false;
        }
    }

    fn get_terminal(&self) -> Self::State {
        (self.env.get_terminal(), 2)
    }
}
