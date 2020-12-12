from gym_snake.register import register

for num_players in ['']:
    for style in ['']:
        for grid_size in ['4x4', '8x8', '16x16']:
            for grid_type in ['']:
                env_id = '-'.join(['Snake', grid_type, grid_size, style, num_players]) + '-v0'.replace('--', '-')
                entry_point = 'gym_snake.envs:' + '_'.join(['Snake', grid_type, grid_size, style, num_players]).replace('--', '-')
                print("register(")
                print("    id='" + env_id + "',")
                print("    entry_point='" + entry_point + "'")
                print(")")

                pass

register(
    id='Snake-4x4-v0',
    entry_point='gym_snake.envs:Snake_4x4'
)

register(
    id='Snake-8x8-v0',
    entry_point='gym_snake.envs:Snake_8x8'
)

register(
    id='Snake-16x16-v0',
    entry_point='gym_snake.envs:Snake_16x16'
)