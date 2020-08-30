use crate::gridworld::GridWorldDefinition;

/// See pg. 132 of <book>
pub fn cliff(width: i32, height: i32) -> GridWorldDefinition {
    let mut definition = GridWorldDefinition::new((width, height), (0, 0), (width - 1, 0), -1.);
    definition.apply_reward((1, 0), (width - 2, 1), -100.);
    definition
}
