use std::cmp;
use std::collections::HashMap;

use crate::environment;

pub type GridIndex = (i32, i32);

fn rect_insert<V: Copy>(
    (x, y): GridIndex,
    (w, h): GridIndex,
    value: V,
    to: &mut HashMap<GridIndex, V>,
) {
    for i in 0..w {
        for j in 0..h {
            to.insert((x + i, y + j), value);
        }
    }
}

/// Affects resultant next state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateEffect {
    BackToStart,
    MoveBy(GridIndex),
}

/// Applies a move in the bounds of the grid space ((0, 0) to (w, h))
fn apply_displacement((dx, dy): GridIndex, (w, h): GridIndex, (x, y): GridIndex) -> GridIndex {
    let (x, y) = (x + dx, y + dy);

    let x = cmp::max(0, cmp::min(x, w - 1));
    let y = cmp::max(0, cmp::min(y, h - 1));

    (x, y)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GridWorldAction {
    Left,
    Right,
    Up,
    Down,
}

impl GridWorldAction {
    pub fn displacement(self) -> GridIndex {
        match self {
            GridWorldAction::Left => (-1, 0),
            GridWorldAction::Right => (1, 0),
            GridWorldAction::Up => (0, 1),
            GridWorldAction::Down => (0, -1),
        }
    }
}

pub struct GridWorldDefinition {
    dimensions: GridIndex,
    start_state: GridIndex,
    end_state: GridIndex,
    default_reward: environment::Reward,
    reward_mask: HashMap<GridIndex, environment::Reward>,
    effect_mask: HashMap<GridIndex, StateEffect>,
}

impl GridWorldDefinition {
    pub fn new(
        dimensions: GridIndex,
        start_state: GridIndex,
        end_state: GridIndex,
        default_reward: environment::Reward,
    ) -> GridWorldDefinition {
        GridWorldDefinition {
            dimensions,
            start_state,
            end_state,
            default_reward,
            reward_mask: HashMap::new(),
            effect_mask: HashMap::new(),
        }
    }

    /// Applies a uniform reward to a rectangle in the state
    pub fn apply_reward(
        &mut self,
        (x, y): GridIndex, // bottom left
        (w, h): GridIndex, // widght, height
        reward: environment::Reward,
    ) {
        rect_insert((x, y), (w, h), reward, &mut self.reward_mask);
    }

    /// Applies a uniform effect to a rectangle in the state
    pub fn apply_effect(
        &mut self,
        (x, y): GridIndex, // bottom left
        (w, h): GridIndex, // widght, height
        effect: StateEffect,
    ) {
        rect_insert((x, y), (w, h), effect, &mut self.effect_mask);
    }

    pub fn world(self) -> GridWorld {
        GridWorld::new(self)
    }
}

pub struct GridWorld {
    definition: GridWorldDefinition,
    current_state: GridIndex,
}

impl GridWorld {
    pub fn new(definition: GridWorldDefinition) -> GridWorld {
        GridWorld {
            current_state: definition.start_state,
            definition,
        }
    }
}

impl environment::MDP for GridWorld {}

impl environment::Environment for GridWorld {
    type Action = GridWorldAction;
    type State = GridIndex;

    fn take_action(&mut self, action: Self::Action) -> Option<(Self::State, environment::Reward)> {
        if self.current_state == self.definition.end_state {
            return None;
        }

        let next_state = apply_displacement(
            action.displacement(),
            self.definition.dimensions,
            self.current_state,
        );

        let reward = *self
            .definition
            .reward_mask
            .get(&next_state)
            .unwrap_or(&self.definition.default_reward);

        let effect_state = match self.definition.effect_mask.get(&next_state) {
            None => next_state,
            Some(effect) => match effect {
                StateEffect::BackToStart => self.definition.start_state,
                StateEffect::MoveBy(displacement) => {
                    apply_displacement(*displacement, self.definition.dimensions, next_state)
                }
            },
        };

        let reward = if effect_state == self.definition.end_state {
            0.
        } else {
            reward
        };

        self.current_state = effect_state;

        Some((effect_state, reward))
    }

    fn available_actions(&self, _: Self::State) -> Vec<Self::Action> {
        vec![
            GridWorldAction::Left,
            GridWorldAction::Right,
            GridWorldAction::Up,
            GridWorldAction::Down,
        ]
    }

    fn current_state(&self) -> Self::State {
        self.current_state
    }

    fn terminated(&self) -> bool {
        self.current_state == self.definition.end_state
    }

    fn is_terminal(&self, state: Self::State) -> bool {
        state == self.definition.end_state
    }

    fn get_terminal(&self) -> Self::State {
        self.definition.end_state
    }
}
