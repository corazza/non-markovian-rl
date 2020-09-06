use std::env;

use reinforcement_learning::gridworld::GridWorld;
use reinforcement_learning::gridworld_definitions;
use reinforcement_learning::learner::TabularLearner;
use reinforcement_learning::learner::{DynaQ, QLearning, Sarsa};
use reinforcement_learning::mdp::MDP;

fn main() {
    let episode_num: u32 = if let Some(arg1) = env::args().nth(1) {
        arg1.parse().unwrap()
    } else {
        println!("Defaulting to 500 episodes each");
        500
    };

    let cliff = gridworld_definitions::cliff(10, 5).world();
    let mut sarsa = Sarsa::<GridWorld>::new(0.1, 0.1, 0.9, 1., cliff.get_terminal());
    let mut ql = QLearning::<GridWorld>::new(0.1, 0.1, 0.9, 1., cliff.get_terminal());
    let mut dynaq = DynaQ::<GridWorld>::new(0.1, 0.1, 1., 50, 1., cliff.get_terminal());

    for i in 0..episode_num {
        if i % (episode_num / 10) == 0 {
            println!("episode {}/{}", i, episode_num);
        }
        let mut cliff_sarsa = gridworld_definitions::cliff(10, 5).world();
        let mut cliff_ql = gridworld_definitions::cliff(10, 5).world();
        let mut cliff_dynaq = gridworld_definitions::cliff(10, 5).world();
        sarsa.episode(&mut cliff_sarsa);
        ql.episode(&mut cliff_ql);
        dynaq.episode(&mut cliff_dynaq);
    }
    println!();

    println!("SARSA sample episode:");
    let mut cliff_sarsa = gridworld_definitions::cliff(10, 5).world();
    sarsa.set_debug(true);
    sarsa.set_epsilon(0.001);
    sarsa.episode(&mut cliff_sarsa);
    println!();

    println!("Q-Learning sample episode:");
    let mut cliff_ql = gridworld_definitions::cliff(10, 5).world();
    ql.set_debug(true);
    ql.set_epsilon(0.001);
    ql.episode(&mut cliff_ql);
    println!();

    println!("Dyna-Q sample episode:");
    let mut cliff_dynaq = gridworld_definitions::cliff(10, 5).world();
    dynaq.set_debug(true);
    dynaq.set_epsilon(0.001);
    dynaq.episode(&mut cliff_dynaq);
    println!();
}
