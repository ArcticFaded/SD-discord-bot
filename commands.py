import argparse
import shlex

SAMPLER_CHOICES = [
    'ddim',
    'k_dpm_2_a',
    'k_dpm_2',
    'k_euler_a',
    'k_euler',
    'k_heun',
    'k_lms',
    'plms',
]

def get_command_parser(default_args={"width": 512, "height": 512, "CFG": 7.5}):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-s', '--steps', default=20, type=int, help='Number of steps')
    parser.add_argument(
        '-W', '--width', type=int, help='Image width, multiple of 64', default=default_args["width"]
    )
    parser.add_argument(
        '-H', '--height', type=int, help='Image height, multiple of 64', default=default_args["height"]
    )
    parser.add_argument(
        '-C',
        '--cfg_scale',
        default=default_args["CFG"],
        type=float,
        help='Classifier free guidance (CFG) scale - higher numbers cause generator to "try" harder.',
    )
    parser.add_argument(
        '-S',
        '--seed',
        type=int,
        default=None,
        help='Image seed; a +ve integer, or use -1 for the previous seed, -2 for the one before that, etc',
    )
    parser.add_argument(
        "-I",
        "--init_image",
        type=str,
        default=None,
        help="Image to base generation off of"
    )
    parser.add_argument(
        "-M",
        "--init_mask",
        type=str,
        default=None,
        help="Mask to restrict generation on"
    )
    parser.add_argument('prompt', nargs=argparse.REMAINDER)

    return parser
