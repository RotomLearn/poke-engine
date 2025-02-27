use crate::generate_instructions::generate_instructions_from_move_pair;
use crate::instruction::StateInstructions;
use crate::observation::generate_observation;
use crate::state::{MoveChoice, Side, SideReference, State};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::thread_rng;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::sync::Arc;
use std::time::Duration;
use tch::{CModule, Device, Tensor};

// Constants for PUCT formula
const C_PUCT: f32 = 2.0;

pub struct NeuralNet {
    model: CModule,
    device: Device,
}

impl NeuralNet {
    pub fn new(model_path: &str, device: Device) -> Result<Self, Box<dyn std::error::Error>> {
        // Explicitly set these before loading the model
        tch::set_num_threads(1);
        
        // Disable internal threading for BLAS operations
        std::env::set_var("OMP_NUM_THREADS", "1");
        std::env::set_var("MKL_NUM_THREADS", "1");
        std::env::set_var("OPENBLAS_NUM_THREADS", "1");
        
        let model = CModule::load(model_path)?;
        Ok(NeuralNet { model, device })
    }

    pub fn evaluate_batch(&self, states: &[State]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
        // Generate observations for all states
        let mut flat_observations = Vec::new();
        let obs_size = generate_observation(&states[0], SideReference::SideOne).len();

        for state in states {
            let obs_s1 = generate_observation(state, SideReference::SideOne);
            let obs_s2 = generate_observation(state, SideReference::SideTwo);
            flat_observations.extend(obs_s1);
            flat_observations.extend(obs_s2);
        }

        // Convert to tensor and reshape for batch processing
        let obs_tensor = Tensor::from_slice(&flat_observations)
            .to_device(self.device)
            .view([states.len() as i64 * 2, obs_size as i64]);
        // Forward pass
        let output = self.model.forward_ts(&[obs_tensor]).unwrap();

        // Split output into policy and value
        let policy_logits = output.slice(1, 0, 3191, 1); // 3191 is policy size
        let value = output.slice(1, 3191, 3192, 1); // Last column is value

        // Convert policy to probabilities
        let policy_probs = policy_logits.softmax(-1, tch::Kind::Float);

        // Extract results for each state
        let mut policies_s1 = Vec::with_capacity(states.len());
        let mut policies_s2 = Vec::with_capacity(states.len());
        let mut values = Vec::with_capacity(states.len());

        for i in 0..states.len() {
            policies_s1.push(Vec::<f32>::try_from(policy_probs.get(2 * (i as i64))).unwrap());
            policies_s2.push(Vec::<f32>::try_from(policy_probs.get(2 * (i as i64) + 1)).unwrap());
            values.push(value.get(2 * (i as i64)).double_value(&[]) as f32);
        }

        (policies_s1, policies_s2, values)
    }

    pub fn evaluate(&self, state: &State) -> (Vec<f32>, Vec<f32>, f32) {
        // Generate observations for both sides
        let obs_s1 = generate_observation(state, SideReference::SideOne);
        let obs_s2 = generate_observation(state, SideReference::SideTwo);

        // Convert to tensors and reshape for batch processing
        let obs_s1_tensor = Tensor::from_slice(&obs_s1)
            .to_device(self.device)
            .view([1, obs_s1.len() as i64]);
        let obs_s2_tensor = Tensor::from_slice(&obs_s2)
            .to_device(self.device)
            .view([1, obs_s2.len() as i64]);

        // Stack for batch processing
        let obs = Tensor::cat(&[obs_s1_tensor, obs_s2_tensor], 0);

        // Forward pass - returns [batch_size, policy_size + 1]
        let output = self.model.forward_ts(&[obs]).unwrap();

        // Split output into policy and value
        // Policy is all but last column, value is last column
        let policy_logits = output.slice(1, 0, 3191, 1); // 3191 is policy size
        let value = output.slice(1, 3191, 3192, 1); // Last column is value

        // Convert policy to probabilities using softmax
        let policy_probs = policy_logits.softmax(-1, tch::Kind::Float);

        // Extract the vectors and value
        let policy_s1: Vec<f32> = Vec::<f32>::try_from(policy_probs.get(0)).unwrap();
        let policy_s2: Vec<f32> = Vec::<f32>::try_from(policy_probs.get(1)).unwrap();
        let value: f32 = value.get(0).double_value(&[]) as f32;

        (policy_s1, policy_s2, value)
    }
}

pub fn get_idx_from_movechoice(side: &Side, move_choice: &MoveChoice) -> usize {
    match move_choice {
        MoveChoice::MoveTera(index) => {
            885 + (side.get_active_immutable().moves[&index].id as usize)
        }
        MoveChoice::Move(index) => side.get_active_immutable().moves[&index].id as usize,
        MoveChoice::Switch(index) => 1770 + (side.pokemon[*index].id as usize),
        MoveChoice::None => 3190,
    }
}

#[derive(Debug)]
pub struct Node {
    pub root: bool,
    pub parent: *mut Node,
    pub children: HashMap<(usize, usize), Vec<Node>>,
    pub times_visited: i64,

    // represents the instructions & s1/s2 moves that led to this node from the parent
    pub instructions: StateInstructions,
    pub s1_choice: usize,
    pub s2_choice: usize,

    // represents the total score and number of visits for this node
    // de-coupled for s1 and s2
    pub s1_options: Vec<MoveNode>,
    pub s2_options: Vec<MoveNode>,

    // New fields for AlphaZero-style MCTS
    pub prior_prob: f32,
    pub value_sum: f32,
    pub value: f32,
}

impl Node {
    fn new(
        s1_options: Vec<MoveChoice>,
        s2_options: Vec<MoveChoice>,
        s1_policy: &[f32],
        s2_policy: &[f32],
        state: &State,
    ) -> Node {
        let s1_options_vec = s1_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
                prior_prob: s1_policy[get_idx_from_movechoice(&state.side_one, &x)],
            })
            .collect();

        let s2_options_vec = s2_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
                prior_prob: s2_policy[get_idx_from_movechoice(&state.side_two, &x)],
            })
            .collect();

        Node {
            root: false,
            parent: std::ptr::null_mut(),
            instructions: StateInstructions::default(),
            times_visited: 0,
            children: HashMap::new(),
            s1_choice: 0,
            s2_choice: 0,
            s1_options: s1_options_vec,
            s2_options: s2_options_vec,
            prior_prob: 0.0,
            value_sum: 0.0,
            value: 0.0,
        }
    }

    fn get_max_depth(&self) -> usize {
        if self.children.is_empty() {
            return 0;
        }

        let mut max_child_depth = 0;
        for (_, children) in &self.children {
            for child in children {
                max_child_depth = max_child_depth.max(child.get_max_depth());
            }
        }
        max_child_depth + 1
    }

    pub fn maximize_puct_for_side(&self, side_map: &[MoveNode]) -> usize {
        let mut choice = 0;
        let mut best_score = f32::MIN;

        for (index, node) in side_map.iter().enumerate() {
            let this_score = node.puct_score(self.times_visited);
            if this_score > best_score {
                best_score = this_score;
                choice = index;
            }
        }
        choice
    }

    pub unsafe fn selection(&mut self, state: &mut State) -> (*mut Node, usize, usize) {
        let return_node = self as *mut Node;

        let s1_mc_index = self.maximize_puct_for_side(&self.s1_options);
        let s2_mc_index = self.maximize_puct_for_side(&self.s2_options);

        let child_vector = self.children.get_mut(&(s1_mc_index, s2_mc_index));
        match child_vector {
            Some(child_vector) => {
                let child_vec_ptr = child_vector as *mut Vec<Node>;
                let chosen_child = self.sample_node(child_vec_ptr);
                state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                (*chosen_child).selection(state)
            }
            None => (return_node, s1_mc_index, s2_mc_index),
        }
    }

    unsafe fn sample_node(&self, move_vector: *mut Vec<Node>) -> *mut Node {
        let mut rng = thread_rng();
        let weights: Vec<f64> = (*move_vector)
            .iter()
            .map(|x| x.instructions.percentage as f64)
            .collect();
        let dist = WeightedIndex::new(weights).unwrap();
        let chosen_node = &mut (*move_vector)[dist.sample(&mut rng)];
        let chosen_node_ptr = chosen_node as *mut Node;
        chosen_node_ptr
    }

    pub unsafe fn expand(
        &mut self,
        state: &mut State,
        s1_move_index: usize,
        s2_move_index: usize,
        network: &NeuralNet,
    ) -> (*mut Node, f32) {
        // Return both node and its value
        let s1_move = &self.s1_options[s1_move_index].move_choice;
        let s2_move = &self.s2_options[s2_move_index].move_choice;

        if (state.battle_is_over() != 0.0 && !self.root)
            || (s1_move == &MoveChoice::None && s2_move == &MoveChoice::None)
        {
            return (self as *mut Node, state.battle_is_over());
        }

        let should_branch_on_damage = self.root || (*self.parent).root;
        let new_instructions =
            generate_instructions_from_move_pair(state, s1_move, s2_move, should_branch_on_damage);

        // Create all states first
        let mut states = Vec::with_capacity(new_instructions.len());
        let mut temp_state = state.clone();

        for instructions in &new_instructions {
            temp_state.apply_instructions(&instructions.instruction_list);
            states.push(temp_state.clone());
            temp_state = state.clone(); // Reset for next iteration
        }

        // Batch evaluate all states
        let (policies_s1, policies_s2, values) = network.evaluate_batch(&states);

        // Create nodes with the evaluated results
        let mut this_pair_vec = Vec::with_capacity(new_instructions.len());

        for ((instructions, state), ((policy_s1, policy_s2), value)) in
            new_instructions.into_iter().zip(states.into_iter()).zip(
                policies_s1
                    .into_iter()
                    .zip(policies_s2.into_iter())
                    .zip(values.into_iter()),
            )
        {
            let (s1_options, s2_options) = state.get_all_options();
            let mut new_node = Node::new(s1_options, s2_options, &policy_s1, &policy_s2, &state);
            new_node.parent = self;
            new_node.instructions = instructions;
            new_node.s1_choice = s1_move_index;
            new_node.s2_choice = s2_move_index;
            new_node.value = value; // Store the value

            this_pair_vec.push(new_node);
        }

        let new_node_ptr = self.sample_node(&mut this_pair_vec);
        state.apply_instructions(&(*new_node_ptr).instructions.instruction_list);
        self.children
            .insert((s1_move_index, s2_move_index), this_pair_vec);

        (new_node_ptr, (*new_node_ptr).value)
    }

    pub unsafe fn backpropagate(&mut self, value: f32, state: &mut State) {
        self.times_visited += 1;
        self.value_sum += value;

        if self.root {
            return;
        }

        let parent_s1_movenode = &mut (*self.parent).s1_options[self.s1_choice];
        parent_s1_movenode.total_score += value;
        parent_s1_movenode.visits += 1;

        let parent_s2_movenode = &mut (*self.parent).s2_options[self.s2_choice];
        parent_s2_movenode.total_score += -value;
        parent_s2_movenode.visits += 1;

        state.reverse_instructions(&self.instructions.instruction_list);
        (*self.parent).backpropagate(value, state);
    }
}

#[derive(Debug)]
pub struct MoveNode {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: i64,
    pub prior_prob: f32,
}

impl MoveNode {
    pub fn puct_score(&self, parent_visits: i64) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY;
        }

        let q_value = self.total_score / self.visits as f32;
        let u_value =
            C_PUCT * self.prior_prob * (parent_visits as f32).sqrt() / (1.0 + self.visits as f32);

        q_value + u_value
    }

    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        self.total_score / self.visits as f32
    }
}

fn do_mcts(root_node: &mut Node, state: &mut State, network: &NeuralNet) {
    let (new_node, s1_move, s2_move) = unsafe { root_node.selection(state) };
    let (new_node, value) = unsafe { (*new_node).expand(state, s1_move, s2_move, network) };
    unsafe { (*new_node).backpropagate(value, state) }
}

#[derive(Clone)]
pub struct MctsSideResultAZ {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: i64,
    pub prior_prob: f32,
}

impl MctsSideResultAZ {
    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        let score = self.total_score / self.visits as f32;
        score
    }
}

pub struct MctsResultAZ {
    pub s1: Vec<MctsSideResultAZ>,
    pub s2: Vec<MctsSideResultAZ>,
    pub iteration_count: i64,
    pub max_depth: usize,
}
pub fn perform_mcts_az(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    network: Arc<NeuralNet>,
) -> MctsResultAZ {
    // Get initial policy for root node
    let (policy_s1, policy_s2, _) = network.evaluate(state);
    let mut root_node = Node::new(
        side_one_options,
        side_two_options,
        &policy_s1,
        &policy_s2,
        state,
    );
    root_node.root = true;

    let start_time = std::time::Instant::now();
    while start_time.elapsed() < max_time {
        for _ in 0..10 {
            do_mcts(&mut root_node, state, &network);
        }
        if root_node.times_visited == 10_000_000 {
            break;
        }
    }
    // let max_depth = root_node.get_max_depth();
    let max_depth = 0;
    MctsResultAZ {
        s1: root_node
            .s1_options
            .iter()
            .map(|v| MctsSideResultAZ {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
                prior_prob: v.prior_prob,
            })
            .collect(),
        s2: root_node
            .s2_options
            .iter()
            .map(|v| MctsSideResultAZ {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
                prior_prob: v.prior_prob,
            })
            .collect(),
        iteration_count: root_node.times_visited,
        max_depth,
    }
}
