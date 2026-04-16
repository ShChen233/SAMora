# Drop-in patch example for train.py
# 1) add: parser.add_argument("--config", type=str, default=None)
# 2) after args = parser.parse_args(), insert:

from config_utils import load_yaml_config, merge_config_into_args

if args.config is not None:
    config_dict = load_yaml_config(args.config)
    args = merge_config_into_args(args, config_dict, parser)

# Then keep the rest of your existing train.py logic unchanged.
