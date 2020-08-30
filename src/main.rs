use reinforcement_learning::gridworld::GridWorld;
use reinforcement_learning::gridworld_definitions;
use reinforcement_learning::learner::TabularLearner;
use reinforcement_learning::learner::{QLearning, Sarsa};
use reinforcement_learning::mdp::MDP;

fn main() {
    let episode_num: u32 = 100000;
    let cliff = gridworld_definitions::cliff(10, 5).world();
    let mut sarsa = Sarsa::<GridWorld>::new(0.1, 0.1, 0.3, 1., cliff.get_terminal());
    let mut ql = QLearning::<GridWorld>::new(0.1, 0.1, 0.3, 1., cliff.get_terminal());

    for i in 0..episode_num {
        if i % (episode_num / 10) == 0 {
            println!("episode {}/{}", i, episode_num);
        }
        let mut cliff_sarsa = gridworld_definitions::cliff(10, 5).world();
        let mut cliff_ql = gridworld_definitions::cliff(10, 5).world();
        sarsa.episode(&mut cliff_sarsa);
        ql.episode(&mut cliff_ql);
    }

    println!("SARSA sample episode:");
    let mut cliff_sarsa = gridworld_definitions::cliff(10, 5).world();
    sarsa.set_debug(true);
    sarsa.set_epsilon(0.001);
    sarsa.episode(&mut cliff_sarsa);

    println!("Q-Learning sample episode:");
    let mut cliff_ql = gridworld_definitions::cliff(10, 5).world();
    ql.set_debug(true);
    ql.set_epsilon(0.001);
    ql.episode(&mut cliff_ql);
}
