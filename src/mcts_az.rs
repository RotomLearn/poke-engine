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
const FORCED_PLAYOUTS_FACTOR: f32 = 2.0; // k=2 as mentioned in the paper
const MIN_POLICY_WEIGHT: f32 = 0.01; // Minimum policy weight to keep a move after pruning
const BASE_DIRICHLET_ALPHA: f32 = 0.3;
const MAX_LEGAL_MOVES: usize = 13; // Maximum number of legal moves in Pokémon

pub struct NeuralNet {
    model: CModule,
    device: Device,
}

// Function to calculate the number of forced playouts for a move
fn calculate_forced_playouts(prior_prob: f32, total_playouts: i64) -> i64 {
    // nforced = k * P(c) * sqrt(sum of all playouts)
    let forced = (FORCED_PLAYOUTS_FACTOR * prior_prob * (total_playouts as f32))
        .sqrt()
        .round() as i64;
    forced.max(1) // Ensure at least 1 forced playout
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
        let batch_size = states.len();
        let obs_size = generate_observation(&states[0], SideReference::SideOne).len();

        // Pre-allocate all buffers at full size to avoid reallocations
        let total_elements = batch_size * 2 * obs_size;
        let mut flat_observations = Vec::with_capacity(total_elements);

        // Pre-allocate mask buffers
        let total_move_elements = batch_size * 2 * 4;
        let total_switch_elements = batch_size * 2 * 6;
        let total_action_elements = batch_size * 2 * 3191;

        let mut all_move_masks = vec![true; total_move_elements];
        let mut all_switch_masks = vec![true; total_switch_elements];
        let mut all_action_masks = vec![true; total_action_elements];

        for (i, state) in states.iter().enumerate() {
            // Generate observations
            let obs_s1 = generate_observation(state, SideReference::SideOne);
            let obs_s2 = generate_observation(state, SideReference::SideTwo);
            flat_observations.extend_from_slice(&obs_s1);
            flat_observations.extend_from_slice(&obs_s2);

            // Get move options for both sides
            let (s1_options, s2_options) = state.get_all_options();

            // Calculate base indices for this state in the flattened arrays
            let s1_move_base = i * 2 * 4;
            let s2_move_base = s1_move_base + 4;
            let s1_switch_base = i * 2 * 6;
            let s2_switch_base = s1_switch_base + 6;
            let s1_action_base = i * 2 * 3191;
            let s2_action_base = s1_action_base + 3191;

            // Process options directly into the pre-allocated arrays
            for option in &s1_options {
                match option {
                    MoveChoice::Move(idx) => {
                        let idx_usize = *idx as usize;
                        let move_id = state.side_one.get_active_immutable().moves[idx].id as usize;
                        all_move_masks[s1_move_base + idx_usize] = false;
                        if move_id < 885 {
                            all_action_masks[s1_action_base + move_id] = false;
                        }
                    }
                    MoveChoice::MoveTera(idx) => {
                        let idx_usize = *idx as usize;
                        let move_id = state.side_one.get_active_immutable().moves[idx].id as usize;
                        all_move_masks[s1_move_base + idx_usize] = false;
                        if move_id < 885 {
                            all_action_masks[s1_action_base + 885 + move_id] = false;
                        }
                    }
                    MoveChoice::Switch(idx) => {
                        let idx_usize = *idx as usize;
                        let pokemon_id = state.side_one.pokemon[*idx].id as usize;
                        all_switch_masks[s1_switch_base + idx_usize] = false;
                        if pokemon_id < 1420 {
                            all_action_masks[s1_action_base + 1770 + pokemon_id] = false;
                        }
                    }
                    _ => {}
                }
            }

            // Process Side 2 options
            for option in &s2_options {
                match option {
                    MoveChoice::Move(idx) => {
                        let idx_usize = *idx as usize;
                        let move_id = state.side_two.get_active_immutable().moves[idx].id as usize;
                        all_move_masks[s2_move_base + idx_usize] = false;
                        if move_id < 885 {
                            all_action_masks[s2_action_base + move_id] = false;
                        }
                    }
                    MoveChoice::MoveTera(idx) => {
                        let idx_usize = *idx as usize;
                        let move_id = state.side_two.get_active_immutable().moves[idx].id as usize;
                        all_move_masks[s2_move_base + idx_usize] = false;
                        if move_id < 885 {
                            all_action_masks[s2_action_base + 885 + move_id] = false;
                        }
                    }
                    MoveChoice::Switch(idx) => {
                        let idx_usize = *idx as usize;
                        let pokemon_id = state.side_two.pokemon[*idx].id as usize;
                        all_switch_masks[s2_switch_base + idx_usize] = false;
                        if pokemon_id < 1420 {
                            all_action_masks[s2_action_base + 1770 + pokemon_id] = false;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Convert directly to tensors
        let obs_tensor = Tensor::from_slice(&flat_observations)
            .to_device(self.device)
            .view([states.len() as i64 * 2, obs_size as i64]);

        let move_mask_tensor = Tensor::from_slice(&all_move_masks)
            .to_device(self.device)
            .view([states.len() as i64 * 2, 4]);

        let switch_mask_tensor = Tensor::from_slice(&all_switch_masks)
            .to_device(self.device)
            .view([states.len() as i64 * 2, 6]);

        let action_mask_tensor = Tensor::from_slice(&all_action_masks)
            .to_device(self.device)
            .view([states.len() as i64 * 2, 3191]);

        // Forward pass
        let output = self
            .model
            .forward_ts(&[
                obs_tensor,
                move_mask_tensor,
                switch_mask_tensor,
                action_mask_tensor,
            ])
            .unwrap();

        // Process output (this part could be optimized further with slicing)
        let policy_logits = output.slice(1, 0, 3191, 1);
        let value = output.slice(1, 3191, 3192, 1);
        let policy_probs = policy_logits.softmax(-1, tch::Kind::Float);

        // Pre-allocate results vectors
        let mut policies_s1 = Vec::with_capacity(states.len());
        let mut policies_s2 = Vec::with_capacity(states.len());
        let mut values = Vec::with_capacity(states.len());

        // Extract results more efficiently using one tensor operation if possible
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

        // Get move options
        let (s1_options, s2_options) = state.get_all_options();

        // Create mask tensors (true = unavailable)
        let mut s1_move_mask = vec![true; 4];
        let mut s1_switch_mask = vec![true; 6];
        let mut s1_action_mask = vec![true; 3191];

        // Fill in availability based on s1_options
        for option in &s1_options {
            match option {
                MoveChoice::Move(idx) => {
                    let move_id = state.side_one.get_active_immutable().moves[idx].id as usize;
                    s1_move_mask[*idx as usize] = false;
                    s1_action_mask[move_id] = false;
                }
                MoveChoice::MoveTera(idx) => {
                    let move_id = state.side_one.get_active_immutable().moves[idx].id as usize;
                    s1_move_mask[*idx as usize] = false;
                    s1_action_mask[885 + move_id] = false;
                }
                MoveChoice::Switch(idx) => {
                    let pokemon_id = state.side_one.pokemon[*idx].id as usize;
                    s1_switch_mask[*idx as usize] = false;
                    s1_action_mask[1770 + pokemon_id] = false;
                }
                _ => {}
            }
        }

        let mut s2_move_mask = vec![true; 4];
        let mut s2_switch_mask = vec![true; 6];
        let mut s2_action_mask = vec![true; 3191];

        for option in &s2_options {
            match option {
                MoveChoice::Move(idx) => {
                    let move_id = state.side_two.get_active_immutable().moves[idx].id as usize;
                    s2_move_mask[*idx as usize] = false;
                    s2_action_mask[move_id] = false;
                }
                MoveChoice::MoveTera(idx) => {
                    let move_id = state.side_two.get_active_immutable().moves[idx].id as usize;
                    s2_move_mask[*idx as usize] = false;
                    s2_action_mask[885 + move_id] = false;
                }
                MoveChoice::Switch(idx) => {
                    let pokemon_id = state.side_two.pokemon[*idx].id as usize;
                    s2_switch_mask[*idx as usize] = false;
                    s2_action_mask[1770 + pokemon_id] = false;
                }
                _ => {}
            }
        }

        // Convert to tensors and reshape for batch processing
        let obs_s1_tensor = Tensor::from_slice(&obs_s1)
            .to_device(self.device)
            .view([1, obs_s1.len() as i64]);
        let obs_s2_tensor = Tensor::from_slice(&obs_s2)
            .to_device(self.device)
            .view([1, obs_s2.len() as i64]);

        let s1_move_mask_tensor = Tensor::from_slice(&s1_move_mask)
            .to_device(self.device)
            .view([1, 4]);
        let s1_switch_mask_tensor = Tensor::from_slice(&s1_switch_mask)
            .to_device(self.device)
            .view([1, 6]);
        let s1_action_mask_tensor = Tensor::from_slice(&s1_action_mask)
            .to_device(self.device)
            .view([1, 3191]);

        let s2_move_mask_tensor = Tensor::from_slice(&s2_move_mask)
            .to_device(self.device)
            .view([1, 4]);
        let s2_switch_mask_tensor = Tensor::from_slice(&s2_switch_mask)
            .to_device(self.device)
            .view([1, 6]);
        let s2_action_mask_tensor = Tensor::from_slice(&s2_action_mask)
            .to_device(self.device)
            .view([1, 3191]);

        // Stack for batch processing
        let obs = Tensor::cat(&[obs_s1_tensor, obs_s2_tensor], 0);
        let move_mask = Tensor::cat(&[s1_move_mask_tensor, s2_move_mask_tensor], 0);
        let switch_mask = Tensor::cat(&[s1_switch_mask_tensor, s2_switch_mask_tensor], 0);
        let action_mask = Tensor::cat(&[s1_action_mask_tensor, s2_action_mask_tensor], 0);

        // Forward pass - returns [batch_size, policy_size + 1]
        let output = self
            .model
            .forward_ts(&[obs, move_mask, switch_mask, action_mask])
            .unwrap();

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

fn sample_gamma(alpha: f32, rng: &mut impl Rng) -> f32 {
    if alpha >= 1.0 {
        // For alpha >= 1, we can use a direct method
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        loop {
            let x = rng.gen::<f32>();
            let v = 1.0 + c * (x - 0.5);
            if v <= 0.0 {
                continue;
            }

            let v = v * v * v;
            let u = rng.gen::<f32>();

            if u < 1.0 - 0.331 * (x - 0.5) * (x - 0.5) {
                return d * v;
            }

            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    } else {
        // For 0 < alpha < 1, use alpha+1 and then transform
        let sample = sample_gamma(alpha + 1.0, rng);
        sample * rng.gen::<f32>().powf(1.0 / alpha)
    }
}

// Function to add Dirichlet noise to a vector of MoveNodes
pub fn add_dirichlet_noise_to_options(
    options: &mut Vec<MoveNode>,
    alpha: f32,
    weight: f32,
    rng: &mut impl Rng,
) {
    if options.is_empty() {
        return;
    }

    // Manually generate Dirichlet noise using Gamma distributions
    let mut gamma_samples = Vec::with_capacity(options.len());
    let mut sum = 0.0;

    // Sample from Gamma distribution with shape parameter alpha
    for _ in 0..options.len() {
        let gamma_sample = sample_gamma(alpha, rng);
        gamma_samples.push(gamma_sample);
        sum += gamma_sample;
    }

    // Normalize to get Dirichlet samples
    let noise: Vec<f32> = gamma_samples.iter().map(|&x| (x / sum) as f32).collect();

    // Mix noise with prior probabilities
    for (i, option) in options.iter_mut().enumerate() {
        option.prior_prob = (1.0 - weight) * option.prior_prob + weight * noise[i];
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

    // Add Dirichlet noise to the root node's prior probabilities
    pub fn add_adaptive_dirichlet_noise(&mut self, params: &AZParams, rng: &mut impl Rng) {
        if !self.root || params.dirichlet_weight <= 0.0 {
            return; // Only apply to root node and if parameters are valid
        }

        // Calculate adaptive alpha for side one
        let s1_legal_move_count = self.s1_options.len();
        let s1_alpha = if params.use_adaptive_alpha && s1_legal_move_count > 0 {
            params.dirichlet_alpha_base * MAX_LEGAL_MOVES as f32 / s1_legal_move_count as f32
        } else {
            params.dirichlet_alpha_base
        };

        // Calculate adaptive alpha for side two
        let s2_legal_move_count = self.s2_options.len();
        let s2_alpha = if params.use_adaptive_alpha && s2_legal_move_count > 0 {
            params.dirichlet_alpha_base * MAX_LEGAL_MOVES as f32 / s2_legal_move_count as f32
        } else {
            params.dirichlet_alpha_base
        };

        // Add noise to side one options with adaptive alpha
        add_dirichlet_noise_to_options(
            &mut self.s1_options,
            s1_alpha,
            params.dirichlet_weight,
            rng,
        );

        // Add noise to side two options with adaptive alpha
        add_dirichlet_noise_to_options(
            &mut self.s2_options,
            s2_alpha,
            params.dirichlet_weight,
            rng,
        );
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

    pub fn maximize_puct_for_side_with_forced(
        &self,
        side_map: &[MoveNode],
        is_root: bool,
    ) -> usize {
        let mut choice = 0;
        let mut best_score = f32::MIN;

        for (index, node) in side_map.iter().enumerate() {
            // For root node, we pass the noised prior for forced playouts
            let noised_prior = if is_root {
                Some(node.prior_prob) // This already includes Dirichlet noise if applied
            } else {
                None
            };

            let this_score = node.puct_score_with_forced(self.times_visited, is_root, noised_prior);
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

    // Modified selection method to use forced playouts
    pub unsafe fn selection_with_forced(&mut self, state: &mut State) -> (*mut Node, usize, usize) {
        let return_node = self as *mut Node;

        // Use forced playouts only at the root
        let is_root = self.root;
        let s1_mc_index = self.maximize_puct_for_side_with_forced(&self.s1_options, is_root);
        let s2_mc_index = self.maximize_puct_for_side_with_forced(&self.s2_options, is_root);

        let child_vector = self.children.get_mut(&(s1_mc_index, s2_mc_index));
        match child_vector {
            Some(child_vector) => {
                let child_vec_ptr = child_vector as *mut Vec<Node>;
                let chosen_child = self.sample_node(child_vec_ptr);
                state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                (*chosen_child).selection_with_forced(state)
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

    pub fn puct_score_with_forced(
        &self,
        parent_visits: i64,
        is_root: bool,
        noised_prior: Option<f32>,
    ) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY;
        }

        // For root children, check if we need forced playouts
        if is_root && noised_prior.is_some() {
            let prior = noised_prior.unwrap();
            let min_visits = calculate_forced_playouts(prior, parent_visits);

            // Force exploration by returning infinity if below min visits
            if self.visits < min_visits {
                return f32::INFINITY;
            }
        }

        // Regular PUCT calculation
        let q_value = self.total_score / self.visits as f32;
        let prior_to_use = noised_prior.unwrap_or(self.prior_prob);
        let u_value =
            C_PUCT * prior_to_use * (parent_visits as f32).sqrt() / (1.0 + self.visits as f32);

        q_value + u_value
    }
}

fn do_mcts(root_node: &mut Node, state: &mut State, network: &NeuralNet) {
    let (new_node, s1_move, s2_move) = unsafe { root_node.selection(state) };
    let (new_node, value) = unsafe { (*new_node).expand(state, s1_move, s2_move, network) };
    unsafe { (*new_node).backpropagate(value, state) }
}

// Modified do_mcts function to use forced playouts
pub fn do_mcts_with_forced(root_node: &mut Node, state: &mut State, network: &NeuralNet) {
    let (new_node, s1_move, s2_move) = unsafe { root_node.selection_with_forced(state) };
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

// Parameters for AlphaZero style MCTS
pub struct AZParams {
    pub dirichlet_alpha_base: f32, // Base alpha value before scaling
    pub use_adaptive_alpha: bool,  // Whether to scale alpha inversely with legal move count
    pub dirichlet_weight: f32,     // How much to weight the noise (e.g., 0.25)
}

impl Default for AZParams {
    fn default() -> Self {
        AZParams {
            dirichlet_alpha_base: BASE_DIRICHLET_ALPHA,
            use_adaptive_alpha: true,
            dirichlet_weight: 0.25,
        }
    }
}

// Function to generate pruned policy targets for training
pub fn get_pruned_policy_targets(moves: &[MctsSideResultAZ]) -> Vec<f32> {
    if moves.is_empty() {
        return Vec::new();
    }

    // Find the move with the most visits
    let best_move = moves.iter().max_by_key(|m| m.visits).unwrap();
    let best_idx = moves
        .iter()
        .position(|m| m.visits == best_move.visits)
        .unwrap();

    // Calculate total visits
    let total_visits: i64 = moves.iter().map(|m| m.visits).sum();

    // Create mutable copies for pruning
    let mut pruned_visits: Vec<i64> = moves.iter().map(|m| m.visits).collect();

    // Calculate the best move's PUCT value components (we'll need this for pruning)
    let best_q = best_move.total_score / best_move.visits as f32;
    let best_prior = best_move.prior_prob;

    // Prune forced playouts for each move
    for (i, move_result) in moves.iter().enumerate() {
        if i == best_idx {
            continue; // Don't prune the best move
        }

        // Skip moves with only one visit - they'll be pruned completely later
        if move_result.visits <= 1 {
            pruned_visits[i] = 0;
            continue;
        }

        // Calculate forced playouts for this move
        let forced = calculate_forced_playouts(move_result.prior_prob, total_visits);
        let actual = move_result.visits;

        // Only prune if we have more than forced playouts
        if actual > forced {
            // Calculate how many we can prune without making this move better than the best move
            let move_q = move_result.total_score / move_result.visits as f32;

            // We need to maintain: PUCT(move) < PUCT(best_move)
            // This is a complex calculation that depends on both Q-values and exploration terms
            // For simplicity, we'll be conservative and just prune to the forced playouts level
            let to_prune = actual - forced;
            pruned_visits[i] = actual - to_prune;
        }
    }

    // Prune moves that have been reduced to just 1 visit
    for visits in pruned_visits.iter_mut() {
        if *visits <= 1 {
            *visits = 0;
        }
    }

    // Calculate the pruned total visits
    let pruned_total: i64 = pruned_visits.iter().sum();
    if pruned_total == 0 {
        // If we've pruned everything (shouldn't happen), just return the raw policy
        return moves.iter().map(|m| m.prior_prob).collect();
    }

    // Normalize to get the pruned policy
    let mut pruned_policy: Vec<f32> = pruned_visits
        .iter()
        .map(|&v| v as f32 / pruned_total as f32)
        .collect();

    // One final check: ensure all moves have at least MIN_POLICY_WEIGHT if they had any visits
    let mut weight_to_redistribute = 0.0;
    for (i, &visits) in pruned_visits.iter().enumerate() {
        if visits == 0 && moves[i].visits > 0 {
            // This move had visits but was pruned
            pruned_policy[i] = MIN_POLICY_WEIGHT;
            weight_to_redistribute += MIN_POLICY_WEIGHT;
        }
    }

    // Redistribute the added minimum weights
    if weight_to_redistribute > 0.0 {
        let scale = (1.0 - weight_to_redistribute) / pruned_policy.iter().sum::<f32>();
        for (i, weight) in pruned_policy.iter_mut().enumerate() {
            if pruned_visits[i] > 0 {
                *weight *= scale;
            }
        }
    }

    pruned_policy
}

pub fn perform_mcts_az(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    network: Arc<NeuralNet>,
) -> MctsResultAZ {
    // Default parameters (no noise)
    perform_mcts_az_with_params(
        state,
        side_one_options,
        side_two_options,
        max_time,
        network,
        None,
    )
}

pub fn perform_mcts_az_with_params(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    network: Arc<NeuralNet>,
    params: Option<AZParams>,
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

    // Apply Dirichlet noise to root if parameters are provided
    if let Some(params) = params {
        let mut rng = thread_rng();
        root_node.add_adaptive_dirichlet_noise(&params, &mut rng);
    }

    let start_time = std::time::Instant::now();
    while start_time.elapsed() < max_time {
        for _ in 0..10 {
            do_mcts(&mut root_node, state, &network);
        }
        if root_node.times_visited == 10_000_000 {
            break;
        }
    }

    let max_depth = 0; // Original code sets this to 0

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

pub fn perform_mcts_az_with_forced(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    network: Arc<NeuralNet>,
    params: Option<AZParams>,
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

    // Apply adaptive Dirichlet noise to root if parameters are provided
    if let Some(params) = params {
        let mut rng = thread_rng();
        root_node.add_adaptive_dirichlet_noise(&params, &mut rng);
    }

    let start_time = std::time::Instant::now();
    while start_time.elapsed() < max_time {
        for _ in 0..10 {
            do_mcts_with_forced(&mut root_node, state, &network);
        }
        if root_node.times_visited == 10_000_000 {
            break;
        }
    }

    // Generate raw results
    let raw_s1 = root_node
        .s1_options
        .iter()
        .map(|v| MctsSideResultAZ {
            move_choice: v.move_choice.clone(),
            total_score: v.total_score,
            visits: v.visits,
            prior_prob: v.prior_prob,
        })
        .collect::<Vec<_>>();

    let raw_s2 = root_node
        .s2_options
        .iter()
        .map(|v| MctsSideResultAZ {
            move_choice: v.move_choice.clone(),
            total_score: v.total_score,
            visits: v.visits,
            prior_prob: v.prior_prob,
        })
        .collect::<Vec<_>>();

    // Apply KataGo's policy target pruning for training policy targets
    let pruned_s1 = get_pruned_policy_targets(&raw_s1);
    let pruned_s2 = get_pruned_policy_targets(&raw_s2);

    // Set the pruned policy weights for each move
    let s1_with_pruned_policy = raw_s1
        .iter()
        .enumerate()
        .map(|(i, v)| MctsSideResultAZ {
            move_choice: v.move_choice.clone(),
            total_score: v.total_score,
            visits: v.visits,
            prior_prob: pruned_s1.get(i).copied().unwrap_or(0.0),
        })
        .collect();

    let s2_with_pruned_policy = raw_s2
        .iter()
        .enumerate()
        .map(|(i, v)| MctsSideResultAZ {
            move_choice: v.move_choice.clone(),
            total_score: v.total_score,
            visits: v.visits,
            prior_prob: pruned_s2.get(i).copied().unwrap_or(0.0),
        })
        .collect();

    MctsResultAZ {
        s1: s1_with_pruned_policy,
        s2: s2_with_pruned_policy,
        iteration_count: root_node.times_visited,
        max_depth: 0,
    }
}
