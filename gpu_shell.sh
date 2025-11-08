#!/bin/bash
python -c "from mlrunner import MLRunner; MLRunner(backend='modal', config_path='lang_config.txt', gpu={'type':'L4'}).shell(interactive=True)"


