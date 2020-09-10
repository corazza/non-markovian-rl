use std::env;

use reinforcement_learning::environment::Environment;
use reinforcement_learning::gridworld::GridWorld;
use reinforcement_learning::gridworld_definitions;
use reinforcement_learning::learner::{DynaQ, NStepSarsa, QLearning, Sarsa};
use reinforcement_learning::learner::{TabularLearner, TabularLearnerConfig};

fn main() {
    let episode_num: u32 = if let Some(arg1) = env::args().nth(1) {
        arg1.parse().unwrap()
    } else {
        println!("Defaulting to 500 episodes each");
        500
    };

    let config = TabularLearnerConfig::new(0.1, 0.1, 0.99, 1.);

    let cliff_terminal = gridworld_definitions::cliff(10, 5).world().get_terminal();

    let mut sarsa = Sarsa::<GridWorld>::new(config.clone(), cliff_terminal);
    let mut ql = QLearning::<GridWorld>::new(config.clone(), cliff_terminal);
    let mut dynaq = DynaQ::<GridWorld>::new(config.clone(), 50, cliff_terminal);
    let mut n_sarsa = NStepSarsa::<GridWorld>::new(5, config.clone(), cliff_terminal);

    for i in 0..episode_num {
        if i % (episode_num / 10) == 0 {
            println!("episode {}/{}", i, episode_num);
        }
        let mut cliff_sarsa = gridworld_definitions::cliff(10, 5).world();
        let mut cliff_ql = gridworld_definitions::cliff(10, 5).world();
        let mut cliff_dynaq = gridworld_definitions::cliff(10, 5).world();
        let mut cliff_n_sarsa = gridworld_definitions::cliff(10, 5).world();
        sarsa.episode(&mut cliff_sarsa);
        ql.episode(&mut cliff_ql);
        dynaq.episode(&mut cliff_dynaq);
        n_sarsa.episode(&mut cliff_n_sarsa);
    }
    println!();

    println!("SARSA sample episode:");
    let mut cliff_sarsa = gridworld_definitions::cliff(10, 5).world();
    sarsa.config.debug = true;
    sarsa.config.epsilon = 0.001;
    sarsa.episode(&mut cliff_sarsa);
    println!();

    println!("Q-Learning sample episode:");
    let mut cliff_ql = gridworld_definitions::cliff(10, 5).world();
    ql.config.debug = true;
    ql.config.epsilon = 0.001;
    ql.episode(&mut cliff_ql);
    println!();

    println!("Dyna-Q sample episode:");
    let mut cliff_dynaq = gridworld_definitions::cliff(10, 5).world();
    dynaq.config.debug = true;
    dynaq.config.epsilon = 0.001;
    dynaq.episode(&mut cliff_dynaq);
    println!();

    println!("N-step SARSA sample episode:");
    let mut cliff_n_sarsa = gridworld_definitions::cliff(10, 5).world();
    n_sarsa.config.debug = true;
    n_sarsa.config.epsilon = 0.001;
    n_sarsa.episode(&mut cliff_n_sarsa);
    println!();
}
