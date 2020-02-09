from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import pickle
import os
import datetime
from blessings import Terminal

from gpustat import __version__
from .core import GPUStatCollection

def update_history(gpu_stats):
    hostname = gpu_stats.hostname

    if not os.path.exists('gpu_history.pkl'):
        pickle.dump({}, open('gpu_history.pkl', 'wb'))

    with open('gpu_history.pkl', 'rb') as f:
        history = pickle.load(f)

        # hostname 추가
        hostname = gpu_stats.hostname
        if hostname not in history:
            history[hostname] = {}

        # 프로세스 정보를 받아와서 유저가 마지막으로 사용한 시각 기록
        for gpu_id, gpu_stat in enumerate(gpu_stats):
            if gpu_id not in history[hostname]:
                history[hostname][gpu_id] = {}
            for p in gpu_stat.processes:
                username = p['username']
                history[hostname][gpu_id][username] = \
                    {'last_used': datetime.datetime.now()}

        # 사용기록이 오래된 유저의 기록 삭제
        for gpu_id in history[hostname].keys():
            cleaned_users = []
            for username, last_used in history[hostname][gpu_id].items():
                used_before = (datetime.datetime.now() - last_used['last_used']).seconds / 3600
                if used_before > 24 * 7:
                    # clean use history longer than 7 days
                    cleaned_users.append(username)
            for username in cleaned_users:
                history[hostname][gpu_id].pop(username)

    with open('gpu_history.pkl', 'wb') as f:
        pickle.dump(history, f)

def print_gpustat(json=False, debug=False, **kwargs):
    '''
    Display the GPU query results into standard output.
    '''
    try:
        gpu_stats = GPUStatCollection.new_query()
    except Exception as e:
        sys.stderr.write('Error on querying NVIDIA devices.'
                         ' Use --debug flag for details\n')
        if debug:
            try:
                import traceback
                traceback.print_exc(file=sys.stderr)
            except Exception:
                # NVMLError can't be processed by traceback:
                #   https://bugs.python.org/issue28603
                # as a workaround, simply re-throw the exception
                raise e
        sys.exit(1)

    if json:
        gpu_stats.print_json(sys.stdout)
    else:
        gpu_stats.print_formatted(sys.stdout, **kwargs)

    update_history(gpu_stats)


def loop_gpustat(interval=1.0, **kwargs):
    term = Terminal()

    with term.fullscreen():
        while 1:
            try:
                query_start = time.time()
                with term.location(0, 0):
                    print_gpustat(eol_char=term.clear_eol + '\n', **kwargs)
                    print(term.clear_eos, end='')
                query_duration = time.time() - query_start
                sleep_duration = interval - query_duration
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
            except KeyboardInterrupt:
                return 0


def main(*argv):
    if not argv:
        argv = list(sys.argv)

    # attach SIGPIPE handler to properly handle broken pipe
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    # arguments to gpustat
    import argparse
    parser = argparse.ArgumentParser()

    parser_color = parser.add_mutually_exclusive_group()
    parser_color.add_argument('--force-color', '--color', action='store_true',
                              help='Force to output with colors')
    parser_color.add_argument('--no-color', action='store_true',
                              help='Suppress colored output')
    parser.add_argument('-a', '--show-all', action='store_true',
                        help='Display all gpu properties above')

    parser.add_argument('-c', '--show-cmd', action='store_true',
                        help='Display cmd name of running process')
    parser.add_argument(
        '-f', '--show-full-cmd', action='store_true',
        help='Display full command and cpu stats of running process'
    )
    parser.add_argument('-u', '--show-user', action='store_true',
                        help='Display username of running process')
    parser.add_argument('-p', '--show-pid', action='store_true',
                        help='Display PID of running process')
    parser.add_argument('-F', '--show-fan-speed', '--show-fan',
                        action='store_true', help='Display GPU fan speed')
    parser.add_argument('--json', action='store_true', default=False,
                        help='Print all the information in JSON format')
    parser.add_argument('-v', '--version', action='version',
                        version=('gpustat %s' % __version__))
    parser.add_argument(
        '-P', '--show-power', nargs='?', const='draw,limit',
        choices=['', 'draw', 'limit', 'draw,limit', 'limit,draw'],
        help='Show GPU power usage or draw (and/or limit)'
    )
    parser.add_argument(
        '-i', '--interval', '--watch', nargs='?', type=float, default=0,
        help='Use watch mode if given; seconds to wait between updates'
    )
    parser.add_argument(
        '--no-header', dest='show_header', action='store_false', default=True,
        help='Suppress header message'
    )
    parser.add_argument(
        '--gpuname-width', type=int, default=16,
        help='The minimum column width of GPU names, defaults to 16'
    )
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='Allow to print additional informations for debugging.'
    )
    args = parser.parse_args(argv[1:])
    if args.show_all:
        args.show_cmd = True
        args.show_user = True
        args.show_pid = True
        args.show_fan_speed = True
        args.show_power = 'draw,limit'
    del args.show_all

    if args.interval is None:  # with default value
        args.interval = 1.0
    if args.interval > 0:
        args.interval = max(0.1, args.interval)
        if args.json:
            sys.stderr.write("Error: --json and --interval/-i "
                             "can't be used together.\n")
            sys.exit(1)

        loop_gpustat(**vars(args))
    else:
        del args.interval
        print_gpustat(**vars(args))


if __name__ == '__main__':
    main(*sys.argv)
