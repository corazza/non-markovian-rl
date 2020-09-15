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

    let normal_steps = 6;

    let print_samples = true;
    let sample_num = 100;

    let mut config = TabularLearnerConfig::new(0.1, 0.05, 0.8, 10.);
    config.debug = false;
    let sarsa_n = 7;
    let dynaq_n = 10; // planning steps

    let terminal = TCorridor::new(normal_steps).get_terminal();
    let m_terminal = wrap_t_corridor(normal_steps).get_terminal();

    let mut sarsa = Sarsa::<TCorridor>::new(config.clone(), terminal);
    let mut m_sarsa = Sarsa::<MWrapper<TCorridor>>::new(config.clone(), m_terminal);
    let mut ql = QLearning::<TCorridor>::new(config.clone(), terminal);
    let mut m_ql = QLearning::<MWrapper<TCorridor>>::new(config.clone(), m_terminal);
    let mut dynaq = DynaQ::<TCorridor>::new(config.clone(), dynaq_n, terminal);
    let mut m_dynaq = DynaQ::<MWrapper<TCorridor>>::new(config.clone(), dynaq_n, m_terminal);
    let mut n_sarsa = NStepSarsa::<TCorridor>::new(sarsa_n, config.clone(), terminal);
    let mut m_n_sarsa = NStepSarsa::<MWrapper<TCorridor>>::new(sarsa_n, config.clone(), m_terminal);

    {
        let mut tasks = vec![
            // Box::new(|| train(|| TCorridor::new(normal_steps), episode_num, &mut sarsa))
            //     as Box<dyn FnMut() + Send>,
            Box::new(|| train(|| wrap_t_corridor(normal_steps), episode_num, &mut m_sarsa))
                as Box<dyn FnMut() + Send>,
            // Box::new(|| train(|| TCorridor::new(normal_steps), episode_num, &mut ql)),
            // Box::new(|| train(|| wrap_t_corridor(normal_steps), episode_num, &mut m_ql)),
            // Box::new(|| train(|| TCorridor::new(normal_steps), episode_num, &mut dynaq)),
            // Box::new(|| train(|| wrap_t_corridor(normal_steps), episode_num, &mut m_dynaq)),
            // Box::new(|| train(|| TCorridor::new(normal_steps), episode_num, &mut n_sarsa))
            //     as Box<dyn FnMut() + Send>,
            Box::new(|| {
                train(
                    || wrap_t_corridor(normal_steps),
                    episode_num,
                    &mut m_n_sarsa,
                )
            }),
        ];

        tasks.par_iter_mut().for_each(|f| f());
    }

    println!();

    println!("Sample M-SARSA episodes: ");
    sample_episodes(
        print_samples,
        || wrap_t_corridor(normal_steps),
        sample_num,
        &mut m_sarsa,
    );
    println!();

    // println!("Sample SARSA episodes: ");
    // sample_episodes(
    //     print_samples,
    //     || TCorridor::new(normal_steps),
    //     sample_num,
    //     &mut sarsa,
    // );
    // println!();

    // println!("Sample M-Q-learning episodes: ");
    // sample_episodes(
    //     print_samples,
    //     || wrap_t_corridor(normal_steps),
    //     sample_num,
    //     &mut m_ql,
    // );
    // println!();

    // println!("Sample Q-learning episodes: ");
    // sample_episodes(
    //     print_samples,
    //     || TCorridor::new(normal_steps),
    //     sample_num,
    //     &mut ql,
    // );
    // println!();

    // println!("Sample M-DynaQ episodes: ");
    // sample_episodes(
    //     print_samples,
    //     || wrap_t_corridor(normal_steps),
    //     10,
    //     &mut m_dynaq,
    // );
    // println!();

    // println!("Sample DynaQ episodes: ");
    // sample_episodes(
    //     print_samples,
    //     || TCorridor::new(normal_steps),
    //     10,
    //     &mut dynaq,
    // );
    // println!();

    println!("Sample M-N-SARSA episodes: ");
    sample_episodes(
        print_samples,
        || wrap_t_corridor(normal_steps),
        sample_num,
        &mut m_n_sarsa,
    );
    println!();

    // println!("Sample N-SARSA episodes: ");
    // sample_episodes(
    //     print_samples,
    //     || TCorridor::new(normal_steps),
    //     sample_num,
    //     &mut n_sarsa,
    // );
    // println!();

    println!("M-SARSA q:");
    print_q(&m_sarsa);

    println!();

    println!("M-N-SARSA q:");
    print_q(&m_n_sarsa);
}

fn print_q<E: Environment, L: TabularLearner<E>>(learner: &L) {
    let mut q: Vec<((E::State, E::Action), Reward)> =
        learner.data().q.clone().into_iter().collect();

    q.sort_by_key(|k| k.0);

    for (k, v) in q {
        println!("{:?} {}", k, v);
    }
}

fn train<E: Environment, F, L: TabularLearner<E>>(new: F, episode_num: usize, learner: &mut L)
where
    F: Fn() -> E,
{
    let report_every = 20;
    // let alpha_changes = 100;
    // let change_by = learner.config().alpha / alpha_changes as f32;

    for i in 0..episode_num {
        if i % (episode_num / report_every) == 0 {
            // learner.config_mut().alpha = learner.config().alpha * 0.85;
            // learner.config_mut().epsilon = learner.config().epsilon * 0.8;
            eprintln!(
                "episode {}/{} (with epsilon={})",
                i,
                episode_num,
                learner.config().epsilon
            );
        }
        let mut env = new();
        learner.episode(&mut env);
    }
}

fn sample_episodes<E: Environment, F, L: TabularLearner<E>>(
    print_samples: bool,
    new: F,
    episode_num: usize,
    learner: &mut L,
) where
    F: Fn() -> E,
{
    let mut total_gain: Reward = 0.0;
    learner.config_mut().epsilon = 0.;
    learner.config_mut().debug = print_samples;

    for i in 0..episode_num {
        if print_samples {
            println!("Sample episode {}/{}", i + 1, episode_num);
        }
        let mut env = new();
        let gain = learner.episode(&mut env);
        println!("Gain: {}", gain);
        total_gain += gain;
    }

    println!("Total gain: {}", total_gain);
}

fn wrap_t_corridor(normal_steps: usize) -> MWrapper<TCorridor> {
    MWrapper::new(TCorridor::new(normal_steps))
}
