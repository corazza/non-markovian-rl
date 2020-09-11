use std::collections::HashMap;
use std::env;

use rayon::prelude::*;
use reinforcement_learning::environment::t_corridor::TCorridor;
use reinforcement_learning::environment::Environment;
use reinforcement_learning::environment::{m_wrapper::MWrapper, Reward};
use reinforcement_learning::learner::{DynaQ, NStepSarsa, QLearning, Sarsa};
use reinforcement_learning::learner::{TabularLearner, TabularLearnerConfig};

fn main() {
    let episode_num: usize = if let Some(arg1) = env::args().nth(1) {
        arg1.parse().unwrap()
    } else {
        println!("Defaulting to 500 episodes");
        500
    };

    let config = TabularLearnerConfig::new(0.1, 0.1, 1.0, 1.);

    let terminal = TCorridor::new().get_terminal();
    let m_terminal = wrap_t_corridor().get_terminal();

    let mut sarsa = Sarsa::<TCorridor>::new(config.clone(), terminal);
    let mut m_sarsa = Sarsa::<MWrapper<TCorridor>>::new(config.clone(), m_terminal);
    let mut ql = QLearning::<TCorridor>::new(config.clone(), terminal);
    let mut m_ql = QLearning::<MWrapper<TCorridor>>::new(config.clone(), m_terminal);
    let mut dynaq = DynaQ::<TCorridor>::new(config.clone(), 10, terminal);
    let mut m_dynaq = DynaQ::<MWrapper<TCorridor>>::new(config.clone(), 10, m_terminal);
    let mut n_sarsa = NStepSarsa::<TCorridor>::new(10, config.clone(), terminal);
    let mut m_n_sarsa = NStepSarsa::<MWrapper<TCorridor>>::new(10, config.clone(), m_terminal);

    {
        let mut tasks = vec![
            Box::new(|| train(TCorridor::new, episode_num, &mut sarsa)) as Box<dyn FnMut() + Send>,
            Box::new(|| train(wrap_t_corridor, episode_num, &mut m_sarsa)),
            Box::new(|| train(TCorridor::new, episode_num, &mut ql)),
            Box::new(|| train(wrap_t_corridor, episode_num, &mut m_ql)),
            Box::new(|| train(TCorridor::new, episode_num, &mut dynaq)),
            Box::new(|| train(wrap_t_corridor, episode_num, &mut m_dynaq)),
            Box::new(|| train(TCorridor::new, episode_num, &mut n_sarsa)),
            Box::new(|| train(wrap_t_corridor, episode_num, &mut m_n_sarsa)),
        ];

        tasks.par_iter_mut().for_each(|f| f());
    }

    println!();

    println!("Sample M-SARSA episodes: ");
    sample_episodes(wrap_t_corridor, 10, &mut m_sarsa);
    println!();

    println!("Sample SARSA episodes: ");
    sample_episodes(TCorridor::new, 10, &mut sarsa);
    println!();

    println!("Sample M-Q-learning episodes: ");
    sample_episodes(wrap_t_corridor, 10, &mut m_ql);
    println!();

    println!("Sample Q-learning episodes: ");
    sample_episodes(TCorridor::new, 10, &mut ql);
    println!();

    println!("Sample M-DynaQ episodes: ");
    sample_episodes(wrap_t_corridor, 10, &mut m_dynaq);
    println!();

    println!("Sample DynaQ episodes: ");
    sample_episodes(TCorridor::new, 10, &mut dynaq);
    println!();

    println!("Sample M-N-SARSA episodes: ");
    sample_episodes(wrap_t_corridor, 10, &mut m_n_sarsa);
    println!();

    println!("Sample N-SARSA episodes: ");
    sample_episodes(TCorridor::new, 10, &mut n_sarsa);
    println!();
}

fn train<E: Environment, F, L: TabularLearner<E>>(new: F, episode_num: usize, learner: &mut L)
where
    F: Fn() -> E,
{
    for i in 0..episode_num {
        if i % (episode_num / 10) == 0 {
            println!("episode {}/{}", i, episode_num);
            learner.config_mut().alpha = learner.config().alpha * 0.65;
            learner.config_mut().epsilon = learner.config().epsilon * 0.65;
        }

        let mut env = new();
        learner.episode(&mut env);
    }
}

fn sample_episodes<E: Environment, F, L: TabularLearner<E>>(
    new: F,
    episode_num: usize,
    learner: &mut L,
) where
    F: Fn() -> E,
{
    let mut total_gain: Reward = 0.0;
    learner.config_mut().epsilon = 0.;
    learner.config_mut().debug = true;

    for i in 0..episode_num {
        println!("Sample episode {}/{}", i + 1, episode_num);
        let mut env = new();
        let gain = learner.episode(&mut env);
        println!("Gain: {}\n", gain);
        total_gain += gain;
    }

    println!("Total gain: {}", total_gain);
}

fn wrap_t_corridor() -> MWrapper<TCorridor> {
    MWrapper::new(TCorridor::new())
}
