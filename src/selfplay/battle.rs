use crate::abilities::ability_on_switch_in;
use crate::generate_instructions::generate_instructions_from_move_pair;
use crate::instruction::{Instruction, StateInstructions};
use crate::items::item_on_switch_in;
use crate::observation::generate_observation;
use crate::selfplay::initialization;
use crate::selfplay::player::{MctsPlayer, Player};
use crate::state::{MoveChoice, PokemonIndex, SideReference, State};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Serialize, Deserialize, Clone)]
pub struct SelfPlayDataPoint {
    observation: Vec<f32>,
    policy: Vec<f32>,
    available_moves: Vec<MoveChoice>,
    outcome: f32,
}

struct GameTurn {
    p1_data: SelfPlayDataPoint,
    p2_data: SelfPlayDataPoint,
}

// Thread-safe file writer
pub struct SharedFileWriter {
    file: Arc<RwLock<File>>,
}

impl SharedFileWriter {
    pub fn new(path: PathBuf) -> io::Result<Self> {
        let file = Arc::new(RwLock::new(
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?,
        ));
        Ok(Self { file })
    }

    pub fn write_line(&self, line: String) -> io::Result<()> {
        let mut file = self.file.write();
        writeln!(file, "{}", line)?;
        Ok(())
    }

    pub fn flush(&self) -> io::Result<()> {
        let mut file = self.file.write();
        file.flush()
    }
}

pub struct SelfPlayRecorder {
    output_writer: Arc<SharedFileWriter>,
    log_file: Option<File>,
    game_history: Vec<GameTurn>,
}

impl SelfPlayRecorder {
    pub fn new(writer: Arc<SharedFileWriter>, log_file: Option<File>) -> Self {
        Self {
            output_writer: writer,
            log_file,
            game_history: Vec::with_capacity(64), // Pre-allocate with reasonable size
        }
    }

    pub fn record_turn(
        &mut self,
        state: &State,
        p1_available_moves: Vec<MoveChoice>,
        p2_available_moves: Vec<MoveChoice>,
        p1_policy: Vec<f32>,
        p2_policy: Vec<f32>,
    ) -> io::Result<()> {
        // Generate observations for both players
        let p1_observation = generate_observation(state, SideReference::SideOne);
        let p2_observation = generate_observation(state, SideReference::SideTwo);

        // // Log observations if we have a log file
        // if let Some(file) = &mut self.log_file {
        //     writeln!(file, "\n=== Player 1 Observation ===")?;
        //     writeln!(
        //         file,
        //         "{}",
        //         inspect_observation(&p1_observation, SideReference::SideOne)
        //     )?;
        //     writeln!(file, "\n=== Player 2 Observation ===")?;
        //     writeln!(
        //         file,
        //         "{}",
        //         inspect_observation(&p2_observation, SideReference::SideTwo)
        //     )?;

        //     // Log policies
        //     writeln!(file, "\n=== Player 1 Policy Vector ===")?;
        //     for (i, weight) in p1_policy.iter().enumerate() {
        //         writeln!(file, "Move {}: {:.4}", i, weight)?;
        //     }
        //     writeln!(file, "\n=== Player 2 Policy Vector ===")?;
        //     for (i, weight) in p2_policy.iter().enumerate() {
        //         writeln!(file, "Move {}: {:.4}", i, weight)?;
        //     }
        // }

        // Create data points
        let p1_data = SelfPlayDataPoint {
            observation: p1_observation,
            policy: p1_policy,
            available_moves: p1_available_moves,
            outcome: 0.0,
        };

        let p2_data = SelfPlayDataPoint {
            observation: p2_observation,
            policy: p2_policy,
            available_moves: p2_available_moves,
            outcome: 0.0,
        };

        // Store turn data
        self.game_history.push(GameTurn { p1_data, p2_data });

        Ok(())
    }

    pub fn finalize_game(&mut self, final_outcome: f32) -> io::Result<()> {
        // Prepare all data points
        let mut data_points = Vec::with_capacity(self.game_history.len() * 2);

        for turn in self.game_history.iter() {
            // Prepare p1's perspective
            let mut p1_data = turn.p1_data.clone();
            p1_data.outcome = final_outcome;
            data_points.push(serde_json::to_string(&p1_data)?);

            // Prepare p2's perspective
            let mut p2_data = turn.p2_data.clone();
            p2_data.outcome = -1.0 * final_outcome;
            data_points.push(serde_json::to_string(&p2_data)?);
        }

        // Write all data points in a single batch
        for data_point in data_points {
            self.output_writer.write_line(data_point)?;
        }

        // Clear history for next game
        self.game_history.clear();

        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.output_writer.flush()
    }
}

impl Drop for SelfPlayRecorder {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            eprintln!("Error flushing recorder on drop: {}", e);
        }
    }
}

pub struct Battle {
    state: State,
    player1: Box<dyn Player + Send>,
    player2: Box<dyn Player + Send>,
    turn_count: u32,
    log_file: Option<File>,
    training_recorder: Option<SelfPlayRecorder>,
}

impl Battle {
    pub fn new_with_training(
        initial_state: State,
        player1: Box<dyn Player + Send>,
        player2: Box<dyn Player + Send>,
        log_path: Option<PathBuf>,
        writer: Arc<SharedFileWriter>,
    ) -> std::io::Result<Self> {
        let log_file = if let Some(path) = log_path {
            Some(
                OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)?,
            )
        } else {
            None
        };

        let training_recorder = Some(SelfPlayRecorder::new(
            writer,
            log_file.as_ref().map(|f| f.try_clone()).transpose()?,
        ));

        Ok(Self {
            state: initial_state,
            player1,
            player2,
            turn_count: 0,
            log_file,
            training_recorder,
        })
    }

    fn log(&mut self, content: &str) -> io::Result<()> {
        if let Some(file) = &mut self.log_file {
            writeln!(file, "{}", content)?;
            file.flush()?;
        }
        Ok(())
    }

    fn log_state(&mut self, prefix: &str) -> io::Result<()> {
        if let Some(file) = &mut self.log_file {
            writeln!(file, "\n{}", prefix)?;
            writeln!(file, "{}", self.state.serialize())?;
            writeln!(file, "\n")?;
            writeln!(file, "{}", self.state.visualize())?;
            file.flush()?;
        }
        Ok(())
    }

    fn format_move(&self, move_choice: &MoveChoice, is_player1: bool) -> String {
        let side = if is_player1 {
            &self.state.side_one
        } else {
            &self.state.side_two
        };

        match move_choice {
            MoveChoice::Move(m) => {
                let active_pokemon = side.get_active_immutable();
                format!("{}", active_pokemon.moves[m].id)
            }
            MoveChoice::MoveTera(m) => {
                let active_pokemon = side.get_active_immutable();
                format!("Terastallize + {}", active_pokemon.moves[m].id)
            }
            MoveChoice::Switch(p) => {
                let pokemon = match p {
                    PokemonIndex::P0 => &side.pokemon.p0,
                    PokemonIndex::P1 => &side.pokemon.p1,
                    PokemonIndex::P2 => &side.pokemon.p2,
                    PokemonIndex::P3 => &side.pokemon.p3,
                    PokemonIndex::P4 => &side.pokemon.p4,
                    PokemonIndex::P5 => &side.pokemon.p5,
                };
                format!("Switch to {}", pokemon.id)
            }
            MoveChoice::None => "None".to_string(),
        }
    }

    fn sample_instructions(
        &mut self,
        move_instructions: Vec<StateInstructions>,
    ) -> Option<StateInstructions> {
        use rand::distributions::WeightedIndex;
        use rand::prelude::*;

        let weights: Vec<f64> = move_instructions
            .iter()
            .map(|inst| inst.percentage as f64)
            .collect();

        let dist = WeightedIndex::new(&weights).unwrap();
        let mut rng = thread_rng();
        let chosen_idx = dist.sample(&mut rng);

        Some(move_instructions[chosen_idx].clone())
    }

    fn generate_initial_switch_in_instructions(&mut self, state: &mut State) -> Vec<Instruction> {
        let mut final_instructions = Vec::new();
        let mut incoming_instructions = StateInstructions::default();

        // Weather/Terrain setters and other simultaneous effects first
        ability_on_switch_in(state, &SideReference::SideOne, &mut incoming_instructions);
        ability_on_switch_in(state, &SideReference::SideTwo, &mut incoming_instructions);
        final_instructions.extend(incoming_instructions.instruction_list.drain(..));

        // Then items
        item_on_switch_in(state, &SideReference::SideOne, &mut incoming_instructions);
        item_on_switch_in(state, &SideReference::SideTwo, &mut incoming_instructions);
        final_instructions.extend(incoming_instructions.instruction_list.drain(..));

        // Reverse instructions since we'll apply them later
        state.reverse_instructions(&final_instructions);

        final_instructions
    }

    pub fn play_game(&mut self) -> io::Result<f32> {
        // Apply initial switch-in instructions
        let mut state_clone = self.state.clone();
        let initial_instructions = self.generate_initial_switch_in_instructions(&mut state_clone);

        if let Some(file) = &mut self.log_file {
            for (j, instruction) in initial_instructions.iter().enumerate() {
                writeln!(file, "  {}: {:?}", j + 1, instruction)?;
            }
        }

        self.state.apply_instructions(&initial_instructions);
        self.log_state("Initial State:")?;
        self.state = State::deserialize(self.state.serialize().as_str());

        while self.state.battle_is_over() == 0.0 {
            self.turn_count += 1;
            self.log(&format!("\n=== Turn {} ===", self.turn_count))?;

            let (side_one_options, side_two_options) = self.state.get_all_options();

            // Format move strings
            let p1_move_strings: Vec<_> = side_one_options
                .iter()
                .map(|m| self.format_move(m, true))
                .collect();
            let p2_move_strings: Vec<_> = side_two_options
                .iter()
                .map(|m| self.format_move(m, false))
                .collect();

            // Log available moves
            if let Some(file) = &mut self.log_file {
                writeln!(file, "\nAvailable moves for Player 1:")?;
                for move_str in &p1_move_strings {
                    writeln!(file, "  {}", move_str)?;
                }
                writeln!(file, "\nAvailable moves for Player 2:")?;
                for move_str in &p2_move_strings {
                    writeln!(file, "  {}", move_str)?;
                }
            }

            // Create state clones and get player references before parallel execution
            let mut state_clone1 = self.state.clone();
            let mut state_clone2 = self.state.clone();
            let player1 = &mut self.player1;
            let player2 = &mut self.player2;

            // Parallel move selection
            let current_turn = self.turn_count;

            let (p1_choice, p2_choice) = rayon::join(
                || player1.choose_move(&mut state_clone1, true, current_turn),
                || player2.choose_move(&mut state_clone2, false, current_turn),
            );

            // Rest of the turn logic...
            let p1_choice_str = self.format_move(&p1_choice.chosen_move, true);
            let p2_choice_str = self.format_move(&p2_choice.chosen_move, false);
            if let Some(file) = &mut self.log_file {
                writeln!(
                    file,
                    "\nPlayer 1 chose: {} ({:?})",
                    p1_choice_str, p1_choice.chosen_move
                )?;
                writeln!(file, "Player 1 move weights: {:?}", p1_choice.move_weights)?;
                writeln!(
                    file,
                    "Player 2 chose: {} ({:?})",
                    p2_choice_str, p2_choice.chosen_move
                )?;
                writeln!(file, "Player 2 move weights: {:?}", p2_choice.move_weights)?;
            }
            // Record training data if we have a recorder
            if let Some(recorder) = &mut self.training_recorder {
                recorder.record_turn(
                    &self.state,
                    side_one_options.clone(),
                    side_two_options.clone(),
                    p1_choice.move_weights.clone(),
                    p2_choice.move_weights.clone(),
                )?;
            }

            // Log chosen moves
            if let Some(file) = &mut self.log_file {
                writeln!(
                    file,
                    "\nPlayer 1 chose: {}\nPlayer 2 chose: {}",
                    p1_choice_str, p2_choice_str
                )?;
            }

            // Generate move instructions
            let move_instructions = generate_instructions_from_move_pair(
                &mut self.state,
                &p1_choice.chosen_move,
                &p2_choice.chosen_move,
                true,
            );

            if let Some(file) = &mut self.log_file {
                writeln!(
                    file,
                    "\nGenerated {} instruction sets",
                    move_instructions.len()
                )?;

                for (i, instructions) in move_instructions.iter().enumerate() {
                    writeln!(
                        file,
                        "\nPossible instruction set {} (probability {:.2}%):",
                        i + 1,
                        instructions.percentage
                    )?;
                    writeln!(
                        file,
                        "Number of instructions: {}",
                        instructions.instruction_list.len()
                    )?;
                    for (j, instruction) in instructions.instruction_list.iter().enumerate() {
                        writeln!(file, "  {}: {:?}", j + 1, instruction)?;
                    }
                }
            }

            // Sample and apply one instruction set based on probabilities
            if let Some(chosen_instructions) = self.sample_instructions(move_instructions) {
                if let Some(file) = &mut self.log_file {
                    writeln!(
                        file,
                        "\nChose instruction set with probability {:.2}%:",
                        chosen_instructions.percentage
                    )?;
                    writeln!(
                        file,
                        "Applying {} instructions:",
                        chosen_instructions.instruction_list.len()
                    )?;
                }

                // Apply all instructions together to maintain proper state
                self.state
                    .apply_instructions(&chosen_instructions.instruction_list);

                if let Some(file) = &mut self.log_file {
                    for (i, instruction) in chosen_instructions.instruction_list.iter().enumerate()
                    {
                        writeln!(file, "Instruction {}: {:?}", i + 1, instruction)?;
                    }
                    self.log_state("State after applying instructions:")?;
                }
                self.state = State::deserialize(self.state.serialize().as_str());
            } else {
                self.log("\nWARNING: No valid instructions generated!")?;
            }

            if let Some(file) = &mut self.log_file {
                writeln!(file, "\nBattle status: {}", self.state.battle_is_over())?;
                file.flush()?;
            }
        }

        let result = self.state.battle_is_over();

        // Finalize the game with the final result if we're recording
        if let Some(recorder) = &mut self.training_recorder {
            recorder.finalize_game(result)?;
        }

        self.log(&format!("\nBattle finished with result: {}", result))?;
        Ok(result)
    }
}
pub fn run_sequential_games(
    num_games: usize,
    writer: Arc<SharedFileWriter>,
    random_teams: &str,
    pokedex: &str,
    movedex_json: &str,
    log_dir: Option<PathBuf>,
) -> io::Result<()> {
    for game_idx in 0..num_games {
        // Initialize new game
        let mut state =
            initialization::initialize_battle_state(random_teams, pokedex, movedex_json);
        state = State::deserialize(state.serialize().as_str());

        let player1 = Box::new(MctsPlayer::new(Duration::from_millis(100)));
        let player2 = Box::new(MctsPlayer::new(Duration::from_millis(100)));

        // Set up logging for this game if requested
        let log_path = log_dir
            .as_ref()
            .map(|dir| dir.join(format!("game_{}.log", game_idx)));

        // Create and run the battle
        let mut battle =
            Battle::new_with_training(state, player1, player2, log_path, Arc::clone(&writer))?;

        println!("Starting game {}", game_idx + 1);
        let result = battle.play_game()?;
        println!("Game {} finished with result: {}", game_idx + 1, result);
    }

    Ok(())
}

/// Results from an evaluation match
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub winner: i8,         // 1 for player1, -1 for player2, 0 for draw
    pub turns: u32,         // Number of turns the battle lasted
    pub duration_ms: u64,   // Time taken for the battle in milliseconds
    pub player1_moves: u32, // Number of moves made by player1
    pub player2_moves: u32, // Number of moves made by player2
}

/// Battle implementation without training data collection
pub struct EvaluationBattle {
    state: State,
    player1: Box<dyn Player + Send>,
    player2: Box<dyn Player + Send>,
    turn_count: u32,
    player1_move_count: u32,
    player2_move_count: u32,
    start_time: Instant,
    log_file: Option<File>,
}

impl EvaluationBattle {
    pub fn new(
        initial_state: State,
        player1: Box<dyn Player + Send>,
        player2: Box<dyn Player + Send>,
        log_path: Option<PathBuf>,
    ) -> std::io::Result<Self> {
        let log_file = if let Some(path) = log_path {
            Some(
                OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)?,
            )
        } else {
            None
        };

        Ok(Self {
            state: initial_state,
            player1,
            player2,
            turn_count: 0,
            player1_move_count: 0,
            player2_move_count: 0,
            start_time: Instant::now(),
            log_file,
        })
    }

    fn log(&mut self, content: &str) -> io::Result<()> {
        if let Some(file) = &mut self.log_file {
            writeln!(file, "{}", content)?;
            file.flush()?;
        }
        Ok(())
    }

    fn log_state(&mut self, prefix: &str) -> io::Result<()> {
        if let Some(file) = &mut self.log_file {
            writeln!(file, "\n{}", prefix)?;
            writeln!(file, "{}", self.state.serialize())?;
            writeln!(file, "\n")?;
            writeln!(file, "{}", self.state.visualize())?;
            file.flush()?;
        }
        Ok(())
    }

    fn format_move(&self, move_choice: &MoveChoice, is_player1: bool) -> String {
        let side = if is_player1 {
            &self.state.side_one
        } else {
            &self.state.side_two
        };

        match move_choice {
            MoveChoice::Move(m) => {
                let active_pokemon = side.get_active_immutable();
                format!("{}", active_pokemon.moves[m].id)
            }
            MoveChoice::MoveTera(m) => {
                let active_pokemon = side.get_active_immutable();
                format!("Terastallize + {}", active_pokemon.moves[m].id)
            }
            MoveChoice::Switch(p) => {
                format!("Switch to Pokemon at index {:?}", p)
            }
            MoveChoice::None => "None".to_string(),
        }
    }

    fn sample_instructions(
        &mut self,
        move_instructions: Vec<StateInstructions>,
    ) -> Option<StateInstructions> {
        use rand::distributions::WeightedIndex;
        use rand::prelude::*;

        let weights: Vec<f64> = move_instructions
            .iter()
            .map(|inst| inst.percentage as f64)
            .collect();

        let dist = WeightedIndex::new(&weights).unwrap();
        let mut rng = thread_rng();
        let chosen_idx = dist.sample(&mut rng);

        Some(move_instructions[chosen_idx].clone())
    }

    fn generate_initial_switch_in_instructions(&mut self, state: &mut State) -> Vec<Instruction> {
        let mut final_instructions = Vec::new();
        let mut incoming_instructions = StateInstructions::default();

        // Weather/Terrain setters and other simultaneous effects first
        ability_on_switch_in(state, &SideReference::SideOne, &mut incoming_instructions);
        ability_on_switch_in(state, &SideReference::SideTwo, &mut incoming_instructions);
        final_instructions.extend(incoming_instructions.instruction_list.drain(..));

        // Then items
        item_on_switch_in(state, &SideReference::SideOne, &mut incoming_instructions);
        item_on_switch_in(state, &SideReference::SideTwo, &mut incoming_instructions);
        final_instructions.extend(incoming_instructions.instruction_list.drain(..));

        // Reverse instructions since we'll apply them later
        state.reverse_instructions(&final_instructions);

        final_instructions
    }

    pub fn play_game(&mut self) -> io::Result<EvaluationResult> {
        self.start_time = Instant::now();

        // Apply initial switch-in instructions
        let mut state_clone = self.state.clone();
        let initial_instructions = self.generate_initial_switch_in_instructions(&mut state_clone);

        if let Some(file) = &mut self.log_file {
            for (j, instruction) in initial_instructions.iter().enumerate() {
                writeln!(file, "  {}: {:?}", j + 1, instruction)?;
            }
        }

        self.state.apply_instructions(&initial_instructions);
        self.log_state("Initial State:")?;

        // Reserialize to ensure clean state
        self.state = State::deserialize(self.state.serialize().as_str());

        while self.state.battle_is_over() == 0.0 {
            self.turn_count += 1;
            self.log(&format!("\n=== Turn {} ===", self.turn_count))?;

            let (side_one_options, side_two_options) = self.state.get_all_options();

            // Format move strings for logging
            let p1_move_strings: Vec<_> = side_one_options
                .iter()
                .map(|m| self.format_move(m, true))
                .collect();
            let p2_move_strings: Vec<_> = side_two_options
                .iter()
                .map(|m| self.format_move(m, false))
                .collect();

            // Log available moves
            if let Some(file) = &mut self.log_file {
                writeln!(file, "\nAvailable moves for Player 1:")?;
                for move_str in &p1_move_strings {
                    writeln!(file, "  {}", move_str)?;
                }
                writeln!(file, "\nAvailable moves for Player 2:")?;
                for move_str in &p2_move_strings {
                    writeln!(file, "  {}", move_str)?;
                }
            }

            // Create state clones and get player references before parallel execution
            let mut state_clone1 = self.state.clone();
            let mut state_clone2 = self.state.clone();
            let player1 = &mut self.player1;
            let player2 = &mut self.player2;

            // Parallel move selection
            let current_turn = self.turn_count;

            let (p1_choice, p2_choice) = rayon::join(
                || player1.choose_move(&mut state_clone1, true, current_turn),
                || player2.choose_move(&mut state_clone2, false, current_turn),
            );

            // Log chosen moves
            let p1_choice_str = self.format_move(&p1_choice.chosen_move, true);
            let p2_choice_str = self.format_move(&p2_choice.chosen_move, false);

            // Update move counters (only count non-None moves)
            if !matches!(p1_choice.chosen_move, MoveChoice::None) {
                self.player1_move_count += 1;
            }
            if !matches!(p2_choice.chosen_move, MoveChoice::None) {
                self.player2_move_count += 1;
            }

            if let Some(file) = &mut self.log_file {
                writeln!(
                    file,
                    "\nPlayer 1 chose: {}\nPlayer 2 chose: {}",
                    p1_choice_str, p2_choice_str
                )?;
            }

            // Generate move instructions
            let move_instructions = generate_instructions_from_move_pair(
                &mut self.state,
                &p1_choice.chosen_move,
                &p2_choice.chosen_move,
                true,
            );

            if let Some(file) = &mut self.log_file {
                writeln!(
                    file,
                    "\nGenerated {} instruction sets",
                    move_instructions.len()
                )?;

                for (i, instructions) in move_instructions.iter().enumerate() {
                    writeln!(
                        file,
                        "\nPossible instruction set {} (probability {:.2}%):",
                        i + 1,
                        instructions.percentage
                    )?;
                    for (j, instruction) in instructions.instruction_list.iter().enumerate() {
                        writeln!(file, "  {}: {:?}", j + 1, instruction)?;
                    }
                }
            }

            // Sample and apply one instruction set based on probabilities
            if let Some(chosen_instructions) = self.sample_instructions(move_instructions) {
                if let Some(file) = &mut self.log_file {
                    writeln!(
                        file,
                        "\nChose instruction set with probability {:.2}%:",
                        chosen_instructions.percentage
                    )?;
                    writeln!(
                        file,
                        "Applying {} instructions:",
                        chosen_instructions.instruction_list.len()
                    )?;
                }

                // Apply all instructions together to maintain proper state
                self.state
                    .apply_instructions(&chosen_instructions.instruction_list);

                if let Some(file) = &mut self.log_file {
                    for (i, instruction) in chosen_instructions.instruction_list.iter().enumerate()
                    {
                        writeln!(file, "Instruction {}: {:?}", i + 1, instruction)?;
                    }
                    self.log_state("State after applying instructions:")?;
                }

                // Reserialize to ensure clean state
                self.state = State::deserialize(self.state.serialize().as_str());
            } else {
                self.log("\nWARNING: No valid instructions generated!")?;
            }

            if let Some(file) = &mut self.log_file {
                writeln!(file, "\nBattle status: {}", self.state.battle_is_over())?;
                file.flush()?;
            }
        }

        let result = self.state.battle_is_over();
        let duration = self.start_time.elapsed();

        self.log(&format!("\nBattle finished with result: {}", result))?;

        // Create and return evaluation result
        let eval_result = EvaluationResult {
            winner: result as i8,
            turns: self.turn_count,
            duration_ms: duration.as_millis() as u64,
            player1_moves: self.player1_move_count,
            player2_moves: self.player2_move_count,
        };

        Ok(eval_result)
    }
}
