from poke_engine import monte_carlo_tree_search, monte_carlo_tree_search_az

from example_state import state


result = monte_carlo_tree_search(state, duration_ms=1000)
print(f"Total Iterations: {result.total_visits}")
print([(i.move_choice, i.total_score, i.visits) for i in result.side_one])
print([(i.move_choice, i.total_score, i.visits) for i in result.side_two])

result = monte_carlo_tree_search_az(state, duration_ms=1000, model_id='/home/mingugaming123/poke-engine/poke-engine-py/python/poke_engine/pokemon_az_quantized.pt')
print(f"Total Iterations: {result.total_visits}")
print([(i.move_choice, i.total_score, i.visits) for i in result.side_one])
print([(i.move_choice, i.total_score, i.visits) for i in result.side_two])