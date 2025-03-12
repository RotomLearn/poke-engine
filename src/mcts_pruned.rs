#![allow(non_snake_case)]
use crate::generate_instructions::generate_instructions_from_move_pair;
use crate::instruction::StateInstructions;
use crate::observation::generate_observation;
use crate::state::Side;
use crate::state::{MoveChoice, SideReference, State};
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
const TOP_K_MOVES: usize = 13; // Number of top moves to consider
const BATCH_SIZE: usize = 64; // Target batch size for neural evaluations

pub struct JointEvaluationBatch {
    pub states: Vec<State>,
    pub node_ptrs: Vec<*mut Node>,
    pub paths: Vec<Vec<StateInstructions>>,
    pub original_state: State,
}

impl JointEvaluationBatch {
    pub fn new(original_state: State, capacity: usize) -> Self {
        JointEvaluationBatch {
            states: Vec::with_capacity(capacity),
            node_ptrs: Vec::with_capacity(capacity),
            paths: Vec::with_capacity(capacity),
            original_state,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.states.len() >= BATCH_SIZE
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.node_ptrs.clear();
        self.paths.clear();
    }
}

pub struct JointNetwork {
    model: CModule,
    device: Device,
}

impl JointNetwork {
    pub fn new(model_path: &str, device: Device) -> Result<Self, Box<dyn std::error::Error>> {
        // Explicitly set these before loading the model
        tch::set_num_threads(1);

        // Disable internal threading for BLAS operations
        std::env::set_var("OMP_NUM_THREADS", "1");
        std::env::set_var("MKL_NUM_THREADS", "1");
        std::env::set_var("OPENBLAS_NUM_THREADS", "1");

        let model = CModule::load(model_path)?;
        Ok(JointNetwork { model, device })
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
        let combined_output = self
            .model
            .forward_ts(&[
                obs_tensor,
                move_mask_tensor,
                switch_mask_tensor,
                action_mask_tensor,
            ])
            .unwrap();

        // Extract policy and value from the combined tensor
        let policy_probs = combined_output.slice(1, 0, 3191, 1);
        let value = combined_output.slice(1, 3191, 3192, 1);

        // Pre-allocate results vectors
        let mut policies_s1 = Vec::with_capacity(states.len());
        let mut policies_s2 = Vec::with_capacity(states.len());
        let mut values = Vec::with_capacity(states.len());

        // Extract results
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

        // Forward pass - returns [batch_size, policy_size]
        let combined_output = self
            .model
            .forward_ts(&[obs, move_mask, switch_mask, action_mask])
            .unwrap();

        // Extract policy and value from the combined tensor
        let policy_probs = combined_output.slice(1, 0, 3191, 1);
        let value = combined_output.slice(1, 3191, 3192, 1);

        // Extract the vectors and value
        let policy_s1: Vec<f32> = Vec::<f32>::try_from(policy_probs.get(0)).unwrap();
        let policy_s2: Vec<f32> = Vec::<f32>::try_from(policy_probs.get(1)).unwrap();
        let value_scalar: f32 = value.get(0).double_value(&[]) as f32;

        (policy_s1, policy_s2, value_scalar)
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

    // Value related fields
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
        // Create MoveNodes from options and policies
        let s1_options_with_policy: Vec<(MoveChoice, f32)> = s1_options
            .iter()
            .map(|x| {
                let idx = get_idx_from_movechoice(&state.side_one, x);
                (x.clone(), s1_policy[idx])
            })
            .collect();

        let s2_options_with_policy: Vec<(MoveChoice, f32)> = s2_options
            .iter()
            .map(|x| {
                let idx = get_idx_from_movechoice(&state.side_two, x);
                (x.clone(), s2_policy[idx])
            })
            .collect();

        // Prune to top-k moves (if needed)
        let pruned_s1 = prune_to_top_k_moves(s1_options_with_policy);
        let pruned_s2 = prune_to_top_k_moves(s2_options_with_policy);

        // Create move nodes from pruned options
        let s1_options_vec = pruned_s1
            .into_iter()
            .map(|(move_choice, prior_prob)| MoveNode {
                move_choice,
                total_score: 0.0,
                visits: 0,
                prior_prob,
            })
            .collect();

        let s2_options_vec = pruned_s2
            .into_iter()
            .map(|(move_choice, prior_prob)| MoveNode {
                move_choice,
                total_score: 0.0,
                visits: 0,
                prior_prob,
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

    pub unsafe fn collect_leaf_node(
        root: *mut Node,
        state: &mut State,
    ) -> (*mut Node, Vec<StateInstructions>) {
        let mut current = root;
        let mut path = Vec::new();

        // Selection phase - traverse to leaf
        loop {
            let s1_choice = (*current).maximize_puct_for_side(&(*current).s1_options);
            let s2_choice = (*current).maximize_puct_for_side(&(*current).s2_options);

            if let Some(children) = (*current).children.get(&(s1_choice, s2_choice)) {
                if !children.is_empty() {
                    let child_ptr =
                        (*current).sample_node(children as *const Vec<Node> as *mut Vec<Node>);
                    path.push((*child_ptr).instructions.clone());
                    state.apply_instructions(&(*child_ptr).instructions.instruction_list);
                    current = child_ptr;
                    continue;
                }
            }
            break;
        }

        // Return the leaf node and the path to it
        (current, path)
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
        network: &JointNetwork,
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

fn do_mcts(root_node: &mut Node, state: &mut State, network: &JointNetwork) {
    let (new_node, s1_move, s2_move) = unsafe { root_node.selection(state) };
    let (new_node, value) = unsafe { (*new_node).expand(state, s1_move, s2_move, network) };
    unsafe { (*new_node).backpropagate(value, state) }
}

// Prune to top-k moves based on policy probabilities
fn prune_to_top_k_moves(moves_with_policy: Vec<(MoveChoice, f32)>) -> Vec<(MoveChoice, f32)> {
    if moves_with_policy.len() <= TOP_K_MOVES {
        return moves_with_policy;
    }

    // Sort by probability (descending)
    let mut sorted_moves = moves_with_policy;
    sorted_moves.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Take top-k moves
    let mut pruned = sorted_moves
        .into_iter()
        .take(TOP_K_MOVES)
        .collect::<Vec<_>>();

    // Normalize probabilities to sum to 1
    let total_prob: f32 = pruned.iter().map(|(_, prob)| prob).sum();
    if total_prob > 0.0 {
        for (_, prob) in pruned.iter_mut() {
            *prob /= total_prob;
        }
    }

    pruned
}

#[derive(Clone)]
pub struct MctsSideResultPruned {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: i64,
    pub prior_prob: f32,
}

impl MctsSideResultPruned {
    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        self.total_score / self.visits as f32
    }
}

pub struct MctsResultPruned {
    pub s1: Vec<MctsSideResultPruned>,
    pub s2: Vec<MctsSideResultPruned>,
    pub iteration_count: i64,
    pub max_depth: usize,
}

pub fn perform_mcts_pruned(
    state: &mut State,
    s1_options: Vec<MoveChoice>,
    s2_options: Vec<MoveChoice>,
    max_time: Duration,
    network: Arc<JointNetwork>,
) -> MctsResultPruned {
    // Get initial policy and all legal moves
    let (policy_s1, policy_s2, _) = network.evaluate(state);

    // Create root node (pruning will happen inside Node::new)
    let mut root_node = Node::new(s1_options, s2_options, &policy_s1, &policy_s2, state);
    root_node.root = true;

    let start_time = std::time::Instant::now();
    let mut iterations = 0;

    while start_time.elapsed() < max_time {
        for _ in 0..10 {
            do_mcts(&mut root_node, state, &network);
            iterations += 1;
        }

        if root_node.times_visited >= 10_000_000 {
            break;
        }
    }

    let max_depth = root_node.get_max_depth();

    MctsResultPruned {
        s1: root_node
            .s1_options
            .iter()
            .map(|v| MctsSideResultPruned {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
                prior_prob: v.prior_prob,
            })
            .collect(),
        s2: root_node
            .s2_options
            .iter()
            .map(|v| MctsSideResultPruned {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
                prior_prob: v.prior_prob,
            })
            .collect(),
        iteration_count: iterations,
        max_depth,
    }
}

fn do_mcts_batch(
    root_node: &mut Node,
    eval_batch: &mut JointEvaluationBatch,
    network: &JointNetwork,
) {
    // Collect a leaf node
    let mut temp_state = eval_batch.original_state.clone();
    let (node_ptr, path) =
        unsafe { Node::collect_leaf_node(root_node as *mut Node, &mut temp_state) };

    // Add to batch
    eval_batch.states.push(temp_state);
    eval_batch.node_ptrs.push(node_ptr);
    eval_batch.paths.push(path);

    // If batch is ready, evaluate and process
    if eval_batch.is_ready() {
        process_joint_evaluation_batch(eval_batch, network);
    }
}

fn process_joint_evaluation_batch(batch: &mut JointEvaluationBatch, network: &JointNetwork) {
    if batch.is_empty() {
        return;
    }

    // Evaluate all states in batch
    let (policies_s1, policies_s2, values) = network.evaluate_batch(&batch.states);

    // Process each result
    for i in 0..batch.states.len() {
        let node_ptr = batch.node_ptrs[i];
        let value = values[i];

        // Get or expand the node
        let (s1_move_index, s2_move_index) = unsafe {
            let s1_idx = (*node_ptr).maximize_puct_for_side(&(*node_ptr).s1_options);
            let s2_idx = (*node_ptr).maximize_puct_for_side(&(*node_ptr).s2_options);
            (s1_idx, s2_idx)
        };

        // Create a state for expansion and backpropagation
        let mut state = batch.original_state.clone();
        for instruction in &batch.paths[i] {
            state.apply_instructions(&instruction.instruction_list);
        }

        // If node needs expansion, expand it with the policy values
        let new_node = unsafe {
            if !(*node_ptr)
                .children
                .contains_key(&(s1_move_index, s2_move_index))
            {
                // Expand using policies from the network
                let s1_move = &(*node_ptr).s1_options[s1_move_index].move_choice;
                let s2_move = &(*node_ptr).s2_options[s2_move_index].move_choice;

                // Generate new instructions
                let should_branch = (*node_ptr).root
                    || ((*node_ptr).parent != std::ptr::null_mut() && (*(*node_ptr).parent).root);
                let instructions = generate_instructions_from_move_pair(
                    &mut state,
                    s1_move,
                    s2_move,
                    should_branch,
                );

                // Create child nodes using policies
                let mut children = Vec::with_capacity(instructions.len());

                for instr in instructions {
                    state.apply_instructions(&instr.instruction_list);
                    let (s1_opts, s2_opts) = state.get_all_options();

                    // Create node with network policies
                    let mut new_node =
                        Node::new(s1_opts, s2_opts, &policies_s1[i], &policies_s2[i], &state);
                    new_node.parent = node_ptr;
                    new_node.instructions = instr.clone();
                    new_node.s1_choice = s1_move_index;
                    new_node.s2_choice = s2_move_index;
                    new_node.value = value;

                    state.reverse_instructions(&instr.instruction_list);
                    children.push(new_node);
                }

                // Add children and select one
                (*node_ptr)
                    .children
                    .insert((s1_move_index, s2_move_index), children);
                let children_vec = (*node_ptr)
                    .children
                    .get_mut(&(s1_move_index, s2_move_index))
                    .unwrap();
                let children_ptr = children_vec as *mut Vec<Node>;
                let child_ptr = (*node_ptr).sample_node(children_ptr);
                state.apply_instructions(&(*child_ptr).instructions.instruction_list);
                child_ptr
            } else {
                // If already expanded, just sample a child
                let children_vec = (*node_ptr)
                    .children
                    .get_mut(&(s1_move_index, s2_move_index))
                    .unwrap();
                let children_ptr = children_vec as *mut Vec<Node>;
                let child_ptr = (*node_ptr).sample_node(children_ptr);
                state.apply_instructions(&(*child_ptr).instructions.instruction_list);
                child_ptr
            }
        };

        // Backpropagate
        unsafe {
            (*new_node).backpropagate(value, &mut state);
        }
    }

    // Clear batch
    batch.clear();
}

pub fn perform_mcts_pruned_batched(
    state: &mut State,
    s1_options: Vec<MoveChoice>,
    s2_options: Vec<MoveChoice>,
    max_time: Duration,
    network: Arc<JointNetwork>,
) -> MctsResultPruned {
    // Get initial policy and all legal moves
    let (policy_s1, policy_s2, _) = network.evaluate(state);

    // Create root node
    let mut root_node = Node::new(s1_options, s2_options, &policy_s1, &policy_s2, state);
    root_node.root = true;

    // Create evaluation batch
    let mut eval_batch = JointEvaluationBatch::new(state.clone(), BATCH_SIZE);

    let start_time = std::time::Instant::now();
    let mut iterations = 0;

    while start_time.elapsed() < max_time {
        for _ in 0..10 {
            do_mcts_batch(&mut root_node, &mut eval_batch, &network);
            iterations += 1;
        }

        // Process any remaining nodes
        if !eval_batch.is_empty() {
            process_joint_evaluation_batch(&mut eval_batch, &network);
        }

        if root_node.times_visited >= 10_000_000 {
            break;
        }
    }

    // Return result
    let max_depth = root_node.get_max_depth();

    MctsResultPruned {
        s1: root_node
            .s1_options
            .iter()
            .map(|v| MctsSideResultPruned {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
                prior_prob: v.prior_prob,
            })
            .collect(),
        s2: root_node
            .s2_options
            .iter()
            .map(|v| MctsSideResultPruned {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
                prior_prob: v.prior_prob,
            })
            .collect(),
        iteration_count: iterations,
        max_depth,
    }
}
