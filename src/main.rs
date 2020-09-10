use std::env;

use reinforcement_learning::environment::gridworld::GridWorld;
use reinforcement_learning::environment::gridworld_definitions;
use reinforcement_learning::environment::t_corridor::{TCorridor, TCorridorAction, TCorridorState};
use reinforcement_learning::environment::Environment;
use reinforcement_learning::learner::{DynaQ, NStepSarsa, QLearning, Sarsa};
use reinforcement_learning::learner::{TabularLearner, TabularLearnerConfig};

fn corridor() {
    let corridor = TCorridor::new();
    let corridor_terminal = corridor.get_terminal();

    let config = TabularLearnerConfig::new(0.1, 0.1, 0.99, 1.);
}

fn train_all_on<E: Environment, F>(
    episode_num: u32,
    config: TabularLearnerConfig,
    new: F,
) -> (Sarsa<E>, QLearning<E>, DynaQ<E>, NStepSarsa<E>)
where
    F: Fn() -> E,
{
    let terminal = new().get_terminal();

    let mut sarsa = Sarsa::<E>::new(config.clone(), terminal);
    let mut ql = QLearning::<E>::new(config.clone(), terminal);
    let mut dynaq = DynaQ::<E>::new(config.clone(), 50, terminal);
    let mut n_sarsa = NStepSarsa::<E>::new(5, config.clone(), terminal);

    for i in 0..episode_num {
        if i % (episode_num / 10) == 0 {
            println!("episode {}/{}", i, episode_num);
        }
        let mut env_sarsa = new();
        let mut env_ql = new();
        let mut env_dynaq = new();
        let mut env_n_sarsa = new();
        sarsa.episode(&mut env_sarsa);
        ql.episode(&mut env_ql);
        dynaq.episode(&mut env_dynaq);
        n_sarsa.episode(&mut env_n_sarsa);
    }

    (sarsa, ql, dynaq, n_sarsa)
}

fn main() {
    let episode_num: u32 = if let Some(arg1) = env::args().nth(1) {
        arg1.parse().unwrap()
    } else {
        println!("Defaulting to 500 episodes each");
        500
    };

    let config = TabularLearnerConfig::new(0.1, 0.1, 1.0, 1.);

    // run_all_on(episode_num, config, || {
    //     gridworld_definitions::cliff(10, 5).world()
    // });

    let (sarsa, ql, dynaq, n_sarsa) = train_all_on(episode_num, config, TCorridor::new);

    fn report_split<L: TabularLearner<TCorridor>>(learner: &L) {
        println!(
            "Up: {}",
            learner.data().q[&(TCorridorState::Split, TCorridorAction::Up)]
        );

        println!(
            "Down: {}",
            learner.data().q[&(TCorridorState::Split, TCorridorAction::Down)]
        );
    }
    println!();

    println!("SARSA:");
    report_split(&sarsa);
    println!();

    println!("Q-Learning:");
    report_split(&ql);
    println!();

    println!("DynaQ:");
    report_split(&dynaq);
    println!();

    println!("N-Step SARSA:");
    report_split(&n_sarsa);
    println!();
}
