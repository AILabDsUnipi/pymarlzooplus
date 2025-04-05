import importlib.util

from pymarlzooplus.envs import REGISTRY_availability as env_REGISTRY_availability


def import_error_pt_butterfly():
    raise ImportError(
        "pettingzoo[butterfly] is not installed! "
        "\nInstall it running: \npip install 'pettingzoo[butterfly]'"
    )


def import_error_pt_atari():
    raise ImportError(
        "pettingzoo[atari] is not installed! "
        "\nInstall it running: \npip install 'pettingzoo[atari]'"
    )


def is_package_installed(package_name):
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None


def atari_rom_error(e):
    # Check if the error message is about the ROM not being installed
    if "Please install roms using AutoROM tool" in str(e):
        if is_package_installed('AutoROM') is False:
            print(
                "The required Atari ROM is not installed. Please install the ROMs using the AutoROM tool."
                "\nYou can install AutoROM by running:"
                "\npip install autorom"
                "\nThen, to automatically download and install Atari ROMs, run:"
                "\nAutoROM -y"
            )
        else:
            raise OSError(
                "The required Atari package 'autorom' is installed, but the Atari ROMs have not been downloaded!."
                "\nRun the following command in your terminal: \nAutoROM -y"
            )
    else:
        raise e


def import_error_pt_classic():
    raise ImportError(
        "pettingzoo[classic] is not installed! "
        "\nInstall it running: \npip install 'pettingzoo[classic]'"
    )


def import_error_pt_mpe():
    raise ImportError(
        "pettingzoo[mpe] is not installed! "
        "\nInstall it running: \npip install 'pettingzoo[mpe]'"
    )


def import_error_pt_sisl():
    raise ImportError(
        "pettingzoo[sisl] is not installed! "
        "\nInstall it running: \npip install 'pettingzoo[sisl]'"
    )


def pettingzoo_make(env_name, kwargs):

    #### Butterfly environments ####

    if env_name == "pistonball_v6":

        if 'continuous' not in kwargs.keys():
            kwargs['continuous'] = False
        assert kwargs['continuous'] is False, "'continuous' argument should be False!"
        if 'n_pistons' in kwargs.keys():
            assert kwargs['n_pistons'] >= 4, \
                "'n_pistons' argument must be greater than or equal to 4!"
            # Otherwise, the game stops almost immediately.

        try:

            from pettingzoo.butterfly import pistonball_v6
            return pistonball_v6.parallel_env(**kwargs)  # Parallel mode

        except ImportError:
            import_error_pt_butterfly()

    elif env_name == "cooperative_pong_v5":
        try:

            from pettingzoo.butterfly import cooperative_pong_v5
            return cooperative_pong_v5.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_butterfly()

    elif env_name == "knights_archers_zombies_v10":

        if 'vector_state' not in kwargs.keys():
            kwargs['vector_state'] = False
        assert kwargs['vector_state'] is False, "'vector_state' argument should be False!"
        if 'use_typemasks' not in kwargs.keys():
            kwargs['use_typemasks'] = False
        assert kwargs['use_typemasks'] is False, "'use_typemasks' argument should be False!"
        if 'sequence_space' not in kwargs.keys():
            kwargs['sequence_space'] = False
        assert kwargs['sequence_space'] is False, "'sequence_space' argument should be False!"

        try:

            from pettingzoo.butterfly import knights_archers_zombies_v10
            return knights_archers_zombies_v10.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_butterfly()

    #### Atari environments ####

    elif env_name == "entombed_cooperative_v3":

        try:

            from pettingzoo.atari import entombed_cooperative_v3
            return entombed_cooperative_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "space_invaders_v2":

        if 'alternating_control' not in kwargs.keys():
            kwargs['alternating_control'] = False
        assert kwargs['alternating_control'] is False, "'alternating_control' should be False!"

        try:

            from pettingzoo.atari import space_invaders_v2
            return space_invaders_v2.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "basketball_pong_v3":

        if 'num_players' not in kwargs.keys():
            kwargs['num_players'] = 2
        assert kwargs['num_players'] in [2, 4], "'num_players' should be 2 or 4!"

        try:

            from pettingzoo.atari import basketball_pong_v3
            return basketball_pong_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "boxing_v2":

        try:

            from pettingzoo.atari import boxing_v2
            return boxing_v2.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "combat_plane_v2":

        if 'game_version' in kwargs.keys():
            assert kwargs['game_version'] in ["jet", "bi-plane"], \
                "'game_version' should be one of following: ['jet', 'bi-plane']!"

        try:
            # There is an inconsistency in PettingZoo Documentation,
            # they say to import like this
            # (top of the page: https://pettingzoo.farama.org/environments/atari/combat_plane/):
            # from pettingzoo.atari import combat_jet_v1
            # then, they use (https://pettingzoo.farama.org/environments/atari/combat_plane/#environment-parameters):
            # combat_plane_v2.env(game_version="jet", guided_missile=True)
            from pettingzoo.atari import combat_plane_v2
            return combat_plane_v2.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "combat_tank_v2":
        try:
            # There is an inconsistency in PettingZoo Documentation,
            # they say to import like this
            # (top of the page: https://pettingzoo.farama.org/environments/atari/combat_tank/):
            # from pettingzoo.atari import combat_tank_v3
            # then, they use (https://pettingzoo.farama.org/environments/atari/combat_tank/#environment-parameters):
            # combat_tank_v2.env(has_maze=True, is_invisible=False, billiard_hit=True)
            from pettingzoo.atari import combat_tank_v2
            return combat_tank_v2.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "double_dunk_v3":
        try:

            from pettingzoo.atari import double_dunk_v3
            return double_dunk_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "entombed_competitive_v3":
        try:

            from pettingzoo.atari import entombed_competitive_v3
            return entombed_competitive_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "flag_capture_v2":
        try:

            from pettingzoo.atari import flag_capture_v2
            return flag_capture_v2.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "foozpong_v3":

        if 'num_players' not in kwargs.keys():
            kwargs['num_players'] = 2
        assert kwargs['num_players'] in [2, 4], "'num_players' should be 2 or 4!"

        try:

            from pettingzoo.atari import foozpong_v3
            return foozpong_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "ice_hockey_v2":
        try:

            from pettingzoo.atari import ice_hockey_v2
            return ice_hockey_v2.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "joust_v3":
        try:

            from pettingzoo.atari import joust_v3
            return joust_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "mario_bros_v3":
        try:

            from pettingzoo.atari import mario_bros_v3
            return mario_bros_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "maze_craze_v3":

        if 'game_version' in kwargs.keys():
            assert kwargs['game_version'] in ["robbers", "race", "capture"], \
                "'game_version' should be one of following: ['robbers', 'race', 'capture']!"
        if 'visibilty_level' in kwargs.keys():
            assert kwargs['visibilty_level'] in [0, 1, 2, 3], \
                "'visibilty_level' should be one of following: [0, 1, 2, 3]!"

        try:

            from pettingzoo.atari import maze_craze_v3
            return maze_craze_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "othello_v3":
        try:

            from pettingzoo.atari import othello_v3
            return othello_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "pong_v3":

        if 'num_players' not in kwargs.keys():
            kwargs['num_players'] = 2
        assert kwargs['num_players'] in [2, 4], "'num_players' should be 2 or 4!"

        try:

            from pettingzoo.atari import pong_v3
            return pong_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "quadrapong_v4":
        try:

            from pettingzoo.atari import quadrapong_v4
            return quadrapong_v4.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "space_war_v2":
        try:

            from pettingzoo.atari import space_war_v2
            return space_war_v2.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "surround_v2":
        try:

            from pettingzoo.atari import surround_v2
            return surround_v2.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "tennis_v3":
        try:

            from pettingzoo.atari import tennis_v3
            return tennis_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "video_checkers_v4":
        try:

            from pettingzoo.atari import video_checkers_v4
            return video_checkers_v4.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "volleyball_pong_v3":

        if 'num_players' not in kwargs.keys():
            kwargs['num_players'] = 4
        assert kwargs['num_players'] in [2, 4], "'num_players' should be 2 or 4!"

        try:
            # There is an inconsistency in PettingZoo Documentation,
            # they say to import like this
            # (top of the page: https://pettingzoo.farama.org/environments/atari/volleyball_pong/):
            # from pettingzoo.atari import volleyball_pong_v2
            # then, they use (https://pettingzoo.farama.org/environments/atari/volleyball_pong/#environment-parameters):
            # volleyball_pong_v3.env(num_players=4)
            from pettingzoo.atari import volleyball_pong_v3
            return volleyball_pong_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "warlords_v3":
        try:

            from pettingzoo.atari import warlords_v3
            return warlords_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    elif env_name == "wizard_of_wor_v3":
        try:

            from pettingzoo.atari import wizard_of_wor_v3
            return wizard_of_wor_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_atari()

        except OSError as e:
            atari_rom_error(e)

    #### Classic environments ####

    elif env_name == "chess_v6":

        if 'max_cycles' in kwargs.keys():
            del kwargs['max_cycles']

        try:
            from pettingzoo.classic import chess_v6
            return chess_v6.env(**kwargs)

        except ImportError:
            import_error_pt_classic()

    elif env_name == "connect_four_v3":

        if 'max_cycles' in kwargs.keys():
            del kwargs['max_cycles']

        try:
            from pettingzoo.classic import connect_four_v3
            return connect_four_v3.env(**kwargs)

        except ImportError:
            import_error_pt_classic()

    elif env_name == "gin_rummy_v4":

        if 'max_cycles' in kwargs.keys():
            del kwargs['max_cycles']

        try:
            from pettingzoo.classic import gin_rummy_v4
            return gin_rummy_v4.env(**kwargs)

        except ImportError:
            import_error_pt_classic()

    elif env_name == "go_v5":

        if 'max_cycles' in kwargs.keys():
            del kwargs['max_cycles']

        try:
            from pettingzoo.classic import go_v5
            return go_v5.env(**kwargs)

        except ImportError:
            import_error_pt_classic()

    elif env_name == "hanabi_v5":

        if 'max_cycles' in kwargs.keys():
            del kwargs['max_cycles']
        if 'players' not in kwargs.keys():
            kwargs['players'] = 2
        if 'hand_size' not in kwargs.keys():
            if kwargs['players'] >= 4:
                kwargs['hand_size'] = 4
            else:
                kwargs['hand_size'] = 5
        else:
            if kwargs['players'] >= 4:
                assert kwargs['hand_size'] == 4, "When 'players'>=4, 'hand_size' should be 4!"
            else:
                assert kwargs['hand_size'] == 5, "When 'players'<4, 'hand_size' should be 5!"

        try:
            from pettingzoo.classic import hanabi_v5
            return hanabi_v5.env(**kwargs)

        except ImportError:
            import_error_pt_classic()

    elif env_name == "leduc_holdem_v4":

        if 'max_cycles' in kwargs.keys():
            del kwargs['max_cycles']

        try:
            from pettingzoo.classic import leduc_holdem_v4
            return leduc_holdem_v4.env(**kwargs)

        except ImportError:
            import_error_pt_classic()

    elif env_name == "rps_v2":

        if 'num_actions' not in kwargs.keys():
            kwargs['num_actions'] = 3
        assert kwargs['num_actions'] in [3, 5], "'num_actions' should be either 3 or 5!"

        try:
            from pettingzoo.classic import rps_v2
            return rps_v2.env(**kwargs)

        except ImportError:
            import_error_pt_classic()

    elif env_name == "texas_holdem_no_limit_v6":

        if 'max_cycles' in kwargs.keys():
            del kwargs['max_cycles']
        if 'num_players' not in kwargs.keys():
            kwargs['num_players'] = 2
        assert kwargs['num_players'] >= 2, "'num_players' should be 2 or greater!"

        try:
            from pettingzoo.classic import texas_holdem_no_limit_v6
            return texas_holdem_no_limit_v6.env(**kwargs)

        except ImportError:
            import_error_pt_classic()

    elif env_name == "texas_holdem_v4":

        if 'max_cycles' in kwargs.keys():
            del kwargs['max_cycles']
        if 'num_players' not in kwargs.keys():
            kwargs['num_players'] = 2
        assert kwargs['num_players'] >= 2, "'num_players' should be 2 or greater!"

        try:
            from pettingzoo.classic import texas_holdem_v4
            return texas_holdem_v4.env(**kwargs)

        except ImportError:
            import_error_pt_classic()

    elif env_name == "tictactoe_v3":

        if 'max_cycles' in kwargs.keys():
            del kwargs['max_cycles']

        try:
            from pettingzoo.classic import tictactoe_v3
            return tictactoe_v3.env(**kwargs)

        except ImportError:
            import_error_pt_classic()

    #### MPE environments ####

    elif env_name == "simple_v3":
        try:

            from pettingzoo.mpe import simple_v3
            return simple_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_mpe()

    elif env_name == "simple_adversary_v3":

        if 'N' not in kwargs.keys():
            kwargs['N'] = 2
        assert kwargs['N'] >= 2, "'N' should be 2 or greater!"

        try:

            from pettingzoo.mpe import simple_adversary_v3
            return simple_adversary_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_mpe()

    elif env_name == "simple_crypto_v3":
        try:

            from pettingzoo.mpe import simple_crypto_v3
            return simple_crypto_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_mpe()

    elif env_name == "simple_push_v3":
        try:

            from pettingzoo.mpe import simple_push_v3
            return simple_push_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_mpe()

    elif env_name == "simple_reference_v3":
        try:

            from pettingzoo.mpe import simple_reference_v3
            return simple_reference_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_mpe()

    elif env_name == "simple_speaker_listener_v4":
        try:

            from pettingzoo.mpe import simple_speaker_listener_v4
            return simple_speaker_listener_v4.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_mpe()

    elif env_name == "simple_spread_v3":

        if 'N' not in kwargs.keys():
            kwargs['N'] = 3
        assert kwargs['N'] >= 3, "'N' should be 2 or greater!"

        try:

            from pettingzoo.mpe import simple_spread_v3
            return simple_spread_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_mpe()

    elif env_name == "simple_tag_v3":

        if 'num_good' not in kwargs.keys():
            kwargs['num_good'] = 1
        assert kwargs['num_good'] >= 0, "'num_good' should be 0 or greater!"
        if 'num_adversaries' not in kwargs.keys():
            kwargs['num_adversaries'] = 3
        assert kwargs['num_adversaries'] >= 0, "'num_adversaries' should be 0 or greater!"
        if 'num_obstacles' not in kwargs.keys():
            kwargs['num_obstacles'] = 2
        assert kwargs['num_obstacles'] >= 0, "'num_obstacles' should be 0 or greater!"

        try:

            from pettingzoo.mpe import simple_tag_v3
            return simple_tag_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_mpe()

    elif env_name == "simple_world_comm_v3":

        if 'num_good' not in kwargs.keys():
            kwargs['num_good'] = 2
        assert kwargs['num_good'] >= 0, "'num_good' should be 0 or greater!"
        if 'num_adversaries' not in kwargs.keys():
            kwargs['num_adversaries'] = 4
        assert kwargs['num_adversaries'] >= 0, "'num_adversaries' should be 0 or greater!"
        if 'num_obstacles' not in kwargs.keys():
            kwargs['num_obstacles'] = 1
        assert kwargs['num_obstacles'] >= 0, "'num_obstacles' should be 0 or greater!"
        if 'num_food' not in kwargs.keys():
            kwargs['num_food'] = 1
        assert kwargs['num_food'] >= 0, "'num_food' should be 0 or greater!"
        if 'num_forests' not in kwargs.keys():
            kwargs['num_forests'] = 1
        assert kwargs['num_forests'] >= 0, "'num_forests' should be 0 or greater!"

        try:

            from pettingzoo.mpe import simple_world_comm_v3
            return simple_world_comm_v3.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_mpe()

    #### SISL environments ####

    elif env_name == "multiwalker_v9":

        if 'n_walkers' not in kwargs.keys():
            kwargs['n_walkers'] = 3
        assert kwargs['n_walkers'] >= 2, "'n_walkers' should be 2 or greater!"
        if 'terrain_length' not in kwargs.keys():
            kwargs['terrain_length'] = 200
        assert kwargs['terrain_length'] >= 50, "'terrain_length' should be 50 or greater!"

        try:

            from pettingzoo.sisl import multiwalker_v9
            return multiwalker_v9.parallel_env(**kwargs)

        except ImportError as e:
            print(e)
            import_error_pt_sisl()

    elif env_name == "pursuit_v4":

        if 'x_size' not in kwargs.keys():
            kwargs['x_size'] = 16
        assert kwargs['x_size'] >= 2, "'x_size' should be 2 or greater!"
        if 'y_size' not in kwargs.keys():
            kwargs['y_size'] = 16
        assert kwargs['y_size'] >= 2, "'y_size' should be 2 or greater!"
        if 'n_evaders' not in kwargs.keys():
            kwargs['n_evaders'] = 30
        assert kwargs['n_evaders'] >= 1, "'n_evaders' should be 1 or greater!"
        if 'n_pursuers' not in kwargs.keys():
            kwargs['n_pursuers'] = 8
        assert kwargs['n_pursuers'] >= 1, "'n_pursuers' should be 1 or greater!"
        if 'obs_range' not in kwargs.keys():
            kwargs['obs_range'] = 7
        assert kwargs['obs_range'] >= 1, "'obs_range' should be 1 or greater!"
        if 'n_catch' not in kwargs.keys():
            kwargs['n_catch'] = 2
        assert kwargs['n_catch'] >= 1, "'n_catch' should be 1 or greater!"

        try:

            from pettingzoo.sisl import pursuit_v4
            return pursuit_v4.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_sisl()

    elif env_name == "waterworld_v4":

        if 'n_evaders' not in kwargs.keys():
            kwargs['n_evaders'] = 5
        assert kwargs['n_evaders'] >= 1, "'n_evaders' should be 1 or greater!"
        if 'n_pursuers' not in kwargs.keys():
            kwargs['n_pursuers'] = 5
        assert kwargs['n_pursuers'] >= 1, "'n_pursuers' should be 1 or greater!"
        if 'n_poisons' not in kwargs.keys():
            kwargs['n_poisons'] = 10
        assert kwargs['n_poisons'] >= 0, "'n_poisons' should be 0 or greater!"
        if 'n_coop' not in kwargs.keys():
            kwargs['n_coop'] = 2
        assert kwargs['n_coop'] >= 0, "'n_coop' should be 0 or greater!"
        if 'n_sensors' not in kwargs.keys():
            kwargs['n_sensors'] = 20
        assert kwargs['n_sensors'] >= 1, "'n_sensors' should be 1 or greater!"
        assert 'obstacle_coord' not in kwargs.keys(), "'obstacle_coord' specification is not supported yet!"

        try:

            from pettingzoo.sisl import waterworld_v4
            return waterworld_v4.parallel_env(**kwargs)

        except ImportError:
            import_error_pt_sisl()

    else:
        raise ValueError(f"Environment '{env_name}' is not supported.")


def check_env_installation(env_name, env_registry, logger):

    if env_name not in list(env_registry.keys()):
        if env_name in env_REGISTRY_availability:
            logger.console_logger.error(
                "\n###########################################"
                f"\nThe requirements for the selected type of environment '{env_name}' have not been installed! "
                "\nPlease follow the installation instruction in the README files."
                "\n###########################################"
            )
        else:
            logger.console_logger.error(
                "\n###########################################"
                f"\nThe selected type of environment '{env_name}' is not supported!"
                f"\nPlease choose one of the following: \n{env_REGISTRY_availability}"
                "\n###########################################"
            )
        exit(0)


