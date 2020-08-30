use reinforcement_learning::gridworld::GridWorld;
use reinforcement_learning::gridworld_definitions;
use reinforcement_learning::learner::Sarsa;

fn main() {
    let mut sarsa = Sarsa::<GridWorld>::new(0.1, 0.1, 0.3, 1.);

    for i in 0..100000 {
        if i % 100 == 0 {
            println!("episode {}", i);
        }
        let mut cliff = gridworld_definitions::cliff(10, 5).world();
        sarsa.episode(&mut cliff);
    }

    let mut cliff = gridworld_definitions::cliff(10, 5).world();
    sarsa.debug = true;
    sarsa.epsilon = 0.001;
    sarsa.episode(&mut cliff);
}
