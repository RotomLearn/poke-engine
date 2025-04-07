use crate::matchup_debug_utils::{MatchupReasoning, compute_matchup_with_reasoning};
use crate::matchup_mcts::{BattleConditions, TeamMatchupCache};
use crate::state::{PokemonIndex, State};
use std::collections::HashMap;
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::Path;
use colored::*;

/// Structure to hold detailed visualizations of matchup analysis
pub struct MatchupVisualizer {
    reasoning_map: HashMap<(usize, usize), MatchupReasoning>,
    cache: TeamMatchupCache,
    results_dir: String,
}

impl MatchupVisualizer {
    /// Create a new matchup visualizer
    pub fn new(cache: TeamMatchupCache) -> Self {
        // Create results directory if it doesn't exist
        let results_dir = "matchup_analysis";
        if !Path::new(results_dir).exists() {
            create_dir_all(results_dir).expect("Failed to create results directory");
        }
        
        MatchupVisualizer {
            reasoning_map: HashMap::new(),
            cache,
            results_dir: results_dir.to_string(),
        }
    }
    
    /// Analyze all matchups with detailed reasoning
    pub fn analyze_all_matchups(&mut self, state: &State) {
        // Get all alive Pokémon
        let s1_alive = self.get_alive_indices(&state.side_one);
        let s2_alive = self.get_alive_indices(&state.side_two);
        
        // Baseline conditions for initial analysis
        let baseline_conditions = BattleConditions {
            hp_brackets: [0, 0], // Full HP
            status: [crate::state::PokemonStatus::NONE, crate::state::PokemonStatus::NONE],
            boosts: [(0, 0, 0, 0, 0), (0, 0, 0, 0, 0)],
            weather: crate::state::Weather::NONE,
            terrain: crate::state::Terrain::NONE,
            trick_room: false,
            terastallized: [false, false],
        };
        
        // Analyze each matchup
        for &s1_idx in &s1_alive {
            for &s2_idx in &s2_alive {
                let (result, reasoning) = compute_matchup_with_reasoning(
                    state, 
                    s1_idx, 
                    s2_idx, 
                    &baseline_conditions
                );
                
                // Store the reasoning
                self.reasoning_map.insert((s1_idx as usize, s2_idx as usize), reasoning);
                
                // Also insert into the cache for traditional analysis
                self.cache.insert(s1_idx as usize, s2_idx as usize, &baseline_conditions, result);
            }
        }
        
        // Save the analysis
        self.save_analysis(state, &s1_alive, &s2_alive);
    }
    
    /// Generate a color-coded HTML visualization of the matchup matrix
    pub fn generate_html_visualization(&self, state: &State) -> std::io::Result<()> {
        let s1_alive = self.get_alive_indices(&state.side_one);
        let s2_alive = self.get_alive_indices(&state.side_two);
        
        let file_path = format!("{}/matchup_visualization.html", self.results_dir);
        let mut file = File::create(file_path)?;
        
        // Write HTML header
        writeln!(file, "<!DOCTYPE html>\n<html>\n<head>\n<title>Pokémon Matchup Analysis</title>")?;
        writeln!(file, "<style>\n  body {{ font-family: Arial, sans-serif; }}\n  table {{ border-collapse: collapse; }}")?;
        writeln!(file, "  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}")?;
        writeln!(file, "  th {{ background-color: #f2f2f2; }}")?;
        writeln!(file, "  .counter {{ background-color: #77dd77; }} /* Strong green */")?;
        writeln!(file, "  .check {{ background-color: #b6e6b6; }} /* Light green */")?;
        writeln!(file, "  .neutral {{ background-color: #f5f5f5; }} /* Light gray */")?;
        writeln!(file, "  .checked {{ background-color: #ffb6c1; }} /* Light red */")?;
        writeln!(file, "  .countered {{ background-color: #ff6961; }} /* Strong red */")?;
        writeln!(file, "  .tooltip {{ position: relative; display: inline-block; }}")?;
        writeln!(file, "  .tooltip .tooltiptext {{ visibility: hidden; width: 300px; background-color: #555; color: #fff; text-align: left; border-radius: 6px; padding: 5px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -150px; opacity: 0; transition: opacity 0.3s; }}")?;
        writeln!(file, "  .tooltip:hover .tooltiptext {{ visibility: visible; opacity: 1; }}")?;
        writeln!(file, "  .reasoning {{ margin-top: 10px; font-style: italic; }}")?;
        writeln!(file, "</style>\n</head>\n<body>")?;
        
        // Add heading
        writeln!(file, "<h1>Pokémon Matchup Analysis</h1>")?;
        
        // Create the table
        writeln!(file, "<table>")?;
        
        // Header row with Side Two Pokémon names
        writeln!(file, "<tr><th></th>")?;
        for &s2_idx in &s2_alive {
            let s2_name = state.side_two.pokemon[s2_idx].id.to_string();
            writeln!(file, "<th>{}</th>", s2_name)?;
        }
        writeln!(file, "</tr>")?;
        
        // Data rows
        for &s1_idx in &s1_alive {
            let s1_name = state.side_one.pokemon[s1_idx].id.to_string();
            writeln!(file, "<tr><th>{}</th>", s1_name)?;
            
            for &s2_idx in &s2_alive {
                if let Some(reasoning) = self.reasoning_map.get(&(s1_idx as usize, s2_idx as usize)) {
                    let css_class = match reasoning.classification {
                        2 => "counter",
                        1 => "check",
                        0 => "neutral",
                        -1 => "checked",
                        -2 => "countered",
                        _ => "neutral",
                    };
                    
                    let matchup_text = match reasoning.classification {
                        2 => "COUNTER",
                        1 => "Check",
                        0 => "Neutral",
                        -1 => "Checked",
                        -2 => "COUNTERED",
                        _ => "Unknown",
                    };
                    
                    // Create tooltip with detailed reasoning
                    let tooltip_text = format!(
                        "{} vs {} - Win Rate: {:.1}%<br><br>Primary Reason: {}<br><br>Key Metrics:<br>• {} moves first<br>• Dmg: {} vs {}<br>• TTK: {} vs {}<br>• Recovery: {} vs {}<br>• OHKO: {:.1}% vs {:.1}%",
                        reasoning.s1_name, reasoning.s2_name, 
                        reasoning.win_percentage * 100.0,
                        reasoning.primary_reason,
                        if reasoning.metrics.s1_moves_first { &reasoning.s1_name } else { &reasoning.s2_name },
                        reasoning.metrics.s1_avg_damage, reasoning.metrics.s2_avg_damage,
                        reasoning.metrics.s1_turns_to_ko, reasoning.metrics.s2_turns_to_ko,
                        reasoning.metrics.s1_recovery_per_turn, reasoning.metrics.s2_recovery_per_turn,
                        reasoning.metrics.s1_ohko_chance * 100.0, reasoning.metrics.s2_ohko_chance * 100.0
                    );
                    
                    writeln!(
                        file,
                        "<td class=\"{}\"><div class=\"tooltip\">{}<span class=\"tooltiptext\">{}</span></div></td>",
                        css_class, matchup_text, tooltip_text
                    )?;
                } else {
                    writeln!(file, "<td>N/A</td>")?;
                }
            }
            
            writeln!(file, "</tr>")?;
        }
        
        writeln!(file, "</table>")?;
        
        // Add legend
        writeln!(file, "<div style=\"margin-top: 20px;\">")?;
        writeln!(file, "<h3>Legend:</h3>")?;
        writeln!(file, "<ul>")?;
        writeln!(file, "<li><span style=\"background-color: #77dd77; padding: 2px 10px;\">COUNTER</span> - Strong advantage (>90% win rate)</li>")?;
        writeln!(file, "<li><span style=\"background-color: #b6e6b6; padding: 2px 10px;\">Check</span> - Favorable advantage (70-90% win rate)</li>")?;
        writeln!(file, "<li><span style=\"background-color: #f5f5f5; padding: 2px 10px;\">Neutral</span> - Even matchup (30-70% win rate)</li>")?;
        writeln!(file, "<li><span style=\"background-color: #ffb6c1; padding: 2px 10px;\">Checked</span> - Unfavorable (10-30% win rate)</li>")?;
        writeln!(file, "<li><span style=\"background-color: #ff6961; padding: 2px 10px;\">COUNTERED</span> - Strong disadvantage (<10% win rate)</li>")?;
        writeln!(file, "</ul>")?;
        writeln!(file, "</div>")?;
        
        // Add instructions
        writeln!(file, "<p><i>Hover over a matchup to see detailed analysis</i></p>")?;
        
        // Close HTML
        writeln!(file, "</body>\n</html>")?;
        
        println!("HTML visualization generated at: {}/matchup_visualization.html", self.results_dir);
        Ok(())
    }
    
    /// Print detailed analysis of a specific matchup
    pub fn print_detailed_matchup(&self, state: &State, s1_idx: PokemonIndex, s2_idx: PokemonIndex) {
        if let Some(reasoning) = self.reasoning_map.get(&(s1_idx as usize, s2_idx as usize)) {
            let s1_name = state.side_one.pokemon[s1_idx].id.to_string();
            let s2_name = state.side_two.pokemon[s2_idx].id.to_string();
            
            // Print header
            println!("{}", "=".repeat(80));
            println!("MATCHUP ANALYSIS: {} vs {}", s1_name.bold(), s2_name.bold());
            println!("{}", "=".repeat(80));
            
            // Print classification
            let classification_str = match reasoning.classification {
                2 => "COUNTER (Strong Favorable)".green().bold(),
                1 => "CHECK (Favorable)".green(),
                0 => "NEUTRAL".white(),
                -1 => "CHECKED (Unfavorable)".red(),
                -2 => "COUNTERED (Strong Unfavorable)".red().bold(),
                _ => "UNKNOWN".white(),
            };
            
            println!("Classification: {} (Win rate: {:.1}%)", 
                classification_str, 
                reasoning.win_percentage * 100.0);
            
            println!("\nPrimary Reason: {}", reasoning.primary_reason.yellow());
            
            // Print key metrics
            println!("\n{}", "KEY METRICS".underline());
            println!("Speed: {} moves first", 
                if reasoning.metrics.s1_moves_first { s1_name.bold() } else { s2_name.bold() });
            
            println!("Damage Output: {} deals {} per turn vs {} deals {} per turn", 
                s1_name, reasoning.metrics.s1_avg_damage.to_string().cyan(),
                s2_name, reasoning.metrics.s2_avg_damage.to_string().cyan());
            
            println!("Turns to KO: {} needs {} turns vs {} needs {} turns", 
                s1_name, reasoning.metrics.s1_turns_to_ko.to_string().cyan(),
                s2_name, reasoning.metrics.s2_turns_to_ko.to_string().cyan());
            
            println!("OHKO Chance: {}: {:.1}% vs {}: {:.1}%", 
                s1_name, (reasoning.metrics.s1_ohko_chance * 100.0).to_string().cyan(),
                s2_name, (reasoning.metrics.s2_ohko_chance * 100.0).to_string().cyan());
            
            println!("Recovery: {}: {} per turn vs {}: {} per turn", 
                s1_name, reasoning.metrics.s1_recovery_per_turn.to_string().cyan(),
                s2_name, reasoning.metrics.s2_recovery_per_turn.to_string().cyan());
            
            // Print reasoning steps
            println!("\n{}", "REASONING STEPS".underline());
            for (i, step) in reasoning.reasoning_steps.iter().enumerate() {
                println!("{}. {}", i+1, step);
            }
            
            println!("{}", "=".repeat(80));
        } else {
            println!("No analysis available for this matchup");
        }
    }
    
    /// Print a complete report of all matchups
    pub fn print_matchup_matrix(&self, state: &State) {
        let s1_alive = self.get_alive_indices(&state.side_one);
        let s2_alive = self.get_alive_indices(&state.side_two);
        
        // Print header
        println!("{}", "=".repeat(80));
        println!("MATCHUP MATRIX OVERVIEW");
        println!("{}", "=".repeat(80));
        
        // Print column headers
        print!("{:<15}", "");
        for &s2_idx in &s2_alive {
            let s2_name = self.truncate_name(&state.side_two.pokemon[s2_idx].id.to_string(), 12);
            print!("{:<12}", s2_name);
        }
        println!();
        
        // Print separator
        print!("{:<15}", "");
        for _ in &s2_alive {
            print!("{:<12}", "-".repeat(12));
        }
        println!();
        
        // Print rows
        for &s1_idx in &s1_alive {
            let s1_name = self.truncate_name(&state.side_one.pokemon[s1_idx].id.to_string(), 15);
            print!("{:<15}", s1_name);
            
            for &s2_idx in &s2_alive {
                if let Some(reasoning) = self.reasoning_map.get(&(s1_idx as usize, s2_idx as usize)) {
                    let result_str = match reasoning.classification {
                        2 => "COUNTER".green().bold(),
                        1 => "Check".green(),
                        0 => "Neutral".white(),
                        -1 => "Checked".red(),
                        -2 => "COUNTERED".red().bold(),
                        _ => "Unknown".white(),
                    };
                    
                    print!("{:<12}", result_str);
                } else {
                    print!("{:<12}", "N/A");
                }
            }
            println!();
        }
        
        println!("\n{}", "Detailed Matchup Analysis".bold());
        println!("For detailed analysis of specific matchups, use print_detailed_matchup()");
        println!("{}", "=".repeat(80));
    }
    
    /// Save all the analysis to files for later review
    fn save_analysis(&self, state: &State, s1_alive: &Vec<PokemonIndex>, s2_alive: &Vec<PokemonIndex>) {
        // Save summary table
        let _ = self.save_summary_table(state, s1_alive, s2_alive);
        
        // Save individual matchup analyses
        let mut combined_file = File::create(format!("{}/all_matchups.txt", self.results_dir))
            .expect("Failed to create combined matchups file");
        
        for &s1_idx in s1_alive {
            for &s2_idx in s2_alive {
                if let Some(reasoning) = self.reasoning_map.get(&(s1_idx as usize, s2_idx as usize)) {
                    // Save to individual file
                    let file_name = format!("{}/{}_{}_vs_{}_{}.txt", 
                        self.results_dir,
                        s1_idx as usize,
                        state.side_one.pokemon[s1_idx].id,
                        s2_idx as usize,
                        state.side_two.pokemon[s2_idx].id
                    );
                    
                    let _ = reasoning.save_to_file(&file_name);
                    
                    // Also append to combined file
                    let _ = reasoning.save_to_file(&format!("{}/all_matchups.txt", self.results_dir));
                }
            }
        }
        
        // Generate HTML visualization
        let _ = self.generate_html_visualization(state);
        
        println!("Analysis saved to '{}' directory", self.results_dir);
    }
    
    /// Save a summary table of all matchups
    fn save_summary_table(&self, state: &State, s1_alive: &Vec<PokemonIndex>, s2_alive: &Vec<PokemonIndex>) -> std::io::Result<()> {
        let file_path = format!("{}/matchup_summary.txt", self.results_dir);
        let mut file = File::create(file_path)?;
        
        // Write header
        writeln!(file, "{}", "=".repeat(80))?;
        writeln!(file, "MATCHUP MATRIX SUMMARY")?;
        writeln!(file, "{}", "=".repeat(80))?;
        
        // Column headers
        write!(file, "{:<15}", "")?;
        for &s2_idx in s2_alive {
            let s2_name = self.truncate_name(&state.side_two.pokemon[s2_idx].id.to_string(), 12);
            write!(file, "{:<12}", s2_name)?;
        }
        writeln!(file)?;
        
        // Separator
        write!(file, "{:<15}", "")?;
        for _ in s2_alive {
            write!(file, "{:<12}", "-".repeat(12))?;
        }
        writeln!(file)?;
        
        // Data rows
        for &s1_idx in s1_alive {
            let s1_name = self.truncate_name(&state.side_one.pokemon[s1_idx].id.to_string(), 15);
            write!(file, "{:<15}", s1_name)?;
            
            for &s2_idx in s2_alive {
                if let Some(reasoning) = self.reasoning_map.get(&(s1_idx as usize, s2_idx as usize)) {
                    let result_str = match reasoning.classification {
                        2 => "COUNTER",
                        1 => "Check",
                        0 => "Neutral",
                        -1 => "Checked",
                        -2 => "COUNTERED",
                        _ => "Unknown",
                    };
                    
                    write!(file, "{:<12}", result_str)?;
                } else {
                    write!(file, "{:<12}", "N/A")?;
                }
            }
            writeln!(file)?;
        }
        
        // Add team counter analysis
        self.write_team_counter_analysis(state, s1_alive, s2_alive, &mut file)?;
        
        // Legend
        writeln!(file, "\nLegend:")?;
        writeln!(file, "COUNTER: Strong advantage (>90% win rate)")?;
        writeln!(file, "Check: Favorable advantage (70-90% win rate)")?;
        writeln!(file, "Neutral: Even matchup (30-70% win rate)")?;
        writeln!(file, "Checked: Unfavorable (10-30% win rate)")?;
        writeln!(file, "COUNTERED: Strong disadvantage (<10% win rate)")?;
        
        Ok(())
    }
    
    /// Add team counter analysis to the summary file
    fn write_team_counter_analysis(
        &self, 
        state: &State, 
        s1_alive: &Vec<PokemonIndex>, 
        s2_alive: &Vec<PokemonIndex>, 
        file: &mut File
    ) -> std::io::Result<()> {
        writeln!(file, "\n{}", "=".repeat(80))?;
        writeln!(file, "TEAM COUNTER ANALYSIS")?;
        writeln!(file, "{}", "=".repeat(80))?;
        
        // For each Side 1 Pokémon
        for &s1_idx in s1_alive {
            let s1_name = &state.side_one.pokemon[s1_idx].id.to_string();
            
            // Find Pokémon this one counters
            let counters: Vec<String> = s2_alive.iter()
                .filter(|&&s2_idx| {
                    if let Some(reasoning) = self.reasoning_map.get(&(s1_idx as usize, s2_idx as usize)) {
                        reasoning.classification >= 2
                    } else {
                        false
                    }
                })
                .map(|&s2_idx| state.side_two.pokemon[s2_idx].id.to_string())
                .collect();
            
            // Find Pokémon that check (but don't counter)
            let checks: Vec<String> = s2_alive.iter()
                .filter(|&&s2_idx| {
                    if let Some(reasoning) = self.reasoning_map.get(&(s1_idx as usize, s2_idx as usize)) {
                        reasoning.classification == 1
                    } else {
                        false
                    }
                })
                .map(|&s2_idx| state.side_two.pokemon[s2_idx].id.to_string())
                .collect();
            
            // Find Pokémon that counter this one
            let countered_by: Vec<String> = s2_alive.iter()
                .filter(|&&s2_idx| {
                    if let Some(reasoning) = self.reasoning_map.get(&(s1_idx as usize, s2_idx as usize)) {
                        reasoning.classification <= -2
                    } else {
                        false
                    }
                })
                .map(|&s2_idx| state.side_two.pokemon[s2_idx].id.to_string())
                .collect();
            
            // Find Pokémon that check this one
            let checked_by: Vec<String> = s2_alive.iter()
                .filter(|&&s2_idx| {
                    if let Some(reasoning) = self.reasoning_map.get(&(s1_idx as usize, s2_idx as usize)) {
                        reasoning.classification == -1
                    } else {
                        false
                    }
                })
                .map(|&s2_idx| state.side_two.pokemon[s2_idx].id.to_string())
                .collect();
            
            writeln!(file, "\n{}: ", s1_name)?;
            
            let counters_str = if counters.is_empty() {
                "None".to_string()
            } else {
                counters.join(", ")
            };
            
            let checks_str = if checks.is_empty() {
                "None".to_string()
            } else {
                checks.join(", ")
            };
            
            let countered_by_str = if countered_by.is_empty() {
                "None".to_string()
            } else {
                countered_by.join(", ")
            };
            
            let checked_by_str = if checked_by.is_empty() {
                "None".to_string()
            } else {
                checked_by.join(", ")
            };
            
            writeln!(file, "  Counters: {}", counters_str)?;
            writeln!(file, "  Checks: {}", checks_str)?;
            writeln!(file, "  Countered by: {}", countered_by_str)?;
            writeln!(file, "  Checked by: {}", checked_by_str)?;
        }
        
        Ok(())
    }
    
    // Helper function to get alive Pokémon indices
    fn get_alive_indices(&self, side: &crate::state::Side) -> Vec<PokemonIndex> {
        let mut indices = Vec::new();
        let mut iter = side.pokemon.into_iter();
        
        while let Some(p) = iter.next() {
            if p.hp > 0 {
                indices.push(iter.pokemon_index.clone());
            }
        }
        
        indices
    }
    
    // Helper function to truncate name
    fn truncate_name(&self, name: &str, max_length: usize) -> String {
        if name.len() <= max_length {
            name.to_string()
        } else {
            format!("{}...", &name[0..(max_length - 3)])
        }
    }
}