import matplotlib.pyplot as plt
import numpy as np
import src.utils as utils
import sys
import os

from pathlib import Path


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

shape_marker = 'x'
shape_linestyle = 'solid'
shape_label = 'Reconstruction'

target_marker = 'd'
target_linestyle = '-'
target_label = 'Target'

reparam_marker = '.'
reparam_linestyle = '-'
reparam_label = r'$\nu$'

momentum_marker = 'x'
momentum_linestyle = 'dotted'
momentum_label = r'$\mathbf{p}$'


def plot_consensus(res_dir: Path):
    vnames = {'momentum': r'$\sum_{i=1}^{N_e}\frac{1}{N_e}\|\bar{p}-p_0^i\|_{L^2}$',
              'theta': r'$\sum_{i=1}^{N_e}\frac{1}{N_e}\|\bar{\theta}-\theta^i\|_{L^2}$'}

    for var in ['momentum', 'theta']:
        if os.path.isfile(res_dir / 'consensuses_{}'.format(var)):
            consensuses = utils.pload(res_dir / 'consensuses_{}'.format(var))

            # plot pickles
            print("Plotting momentum consensuses...")
            ax = plt.subplot(111)
            ax.set_xlim(left=1, right=len(consensuses))
            plt.grid()
            plt.xlabel('Iteration $n$')
            plt.ylabel(vnames[var])
            plt.plot(range(1, len(consensuses) + 1), consensuses)
            plt.savefig(res_dir / 'consensuses_{}.pdf'.format(var))
            plt.clf()


def plot_errors(res_dir: Path):
    eta = -1
    params_path = res_dir / 'inverse_problem_parameters.log'
    if os.path.isfile(params_path):
        params = open(params_path, "r")
        for line in params:
            if line.startswith("eta:"):
                eta = float(line.split()[1])
                break

    errors_path = res_dir / 'errors'
    if os.path.isfile(errors_path):
        print(r'Plotting errors...')
        plt.figure()
        errors = utils.pload(errors_path)
        plt.grid()
        plt.xlabel('Iteration $k$')
        plt.ylabel(r'Error level')
        plt.semilogy(range(1, len(errors) + 1), errors, label=r'Data misfit')
        plt.axhline(y=eta, linestyle=':', color='red', label=r'Noise level')
        plt.legend(loc='best')
        plt.savefig(res_dir / 'data_misfit.pdf')
        plt.clf()

    errors_path = res_dir / 'relative_error_momentum'
    if os.path.isfile(errors_path):
        print(r'Plotting relative momentum error...')
        plt.figure()
        errors = utils.pload(errors_path)
        plt.grid()
        plt.xlabel('Iteration $k$')
        plt.ylabel(r'Relative error')
        plt.semilogy(range(1, len(errors) + 1), errors)
        plt.savefig(res_dir / 'relative_error_momentum.pdf')
        plt.clf()

    errors_path = res_dir / 'relative_error_param'
    if os.path.isfile(errors_path):
        print(r'Plotting relative parameterisation error...')
        plt.figure()
        errors = utils.pload(errors_path)
        plt.grid()
        plt.xlabel('Iteration $k$')
        plt.ylabel(r'Relative error')
        plt.semilogy(range(1, len(errors) + 1), errors)
        plt.savefig(res_dir / 'relative_error_param.pdf')
        plt.clf()


def plot_shape_means(res_dir: Path):
    # count pickles
    num_q_means = 0
    prefix = "q_mean_iter="
    for fn in os.listdir(res_dir):
        if fn.startswith(prefix):
            num_q_means += 1
    print("Number of q_mean files:", num_q_means)

    # load pickles
    q_means = []
    for i in range(num_q_means):
        q_mean = utils.pload(res_dir / (prefix + str(i)))
        q_means.append(q_mean)

    target = utils.pload(res_dir / "target")
    target = np.append(target, [target[0, :]], axis=0)

    # plot pickles
    print("Plotting all means together...")
    plt.figure()
    plt.grid()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    j = 0
    shapes = []
    while j < num_q_means:
        shape = np.append(q_means[j], [q_means[j][0, :]], axis=0)
        plt.plot(shape[:, 0], shape[:, 1])
        j += 1
        shapes.append(shape)

    plt.plot(target[:, 0], target[:, 1], marker=target_marker, linestyle=target_linestyle, label=target_label)
    plt.legend(loc='best')
    plt.savefig(res_dir / 'shape_means.pdf')
    plt.close()

    print("Plotting mean progression...")
    for iteration, shape in enumerate(shapes):
        plt.figure()
        plt.grid()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.plot(shape[:, 0], shape[:, 1], marker=shape_marker, linestyle=shape_linestyle, label=shape_label)
        plt.plot(target[:, 0], target[:, 1], marker=target_marker, linestyle=target_linestyle, label=target_label)
        plt.legend(loc='best')
        plt.savefig(res_dir / f"shape_iter={iteration}.pdf")
        plt.close()


def plot_mismatch(res_dir: Path):
    # count pickles
    num_q_means = 0
    prefix = "mismatch_iter="
    for fn in os.listdir(res_dir):
        if fn.startswith(prefix):
            num_q_means += 1
    print("Number of mismatch files:", num_q_means)

    # load pickles
    q_means = []
    for i in range(num_q_means):
        q_mean = utils.pload(res_dir / (prefix + str(i)))
        q_means.append(q_mean)

    # plot pickles
    print("Plotting means...")
    plt.grid()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    j = 0
    while j < num_q_means:
        shape = q_means[j]
        shape.shape = (shape.size // 2, 2)
        shape = np.append(q_means[j], [q_means[j][0, :]], axis=0)
        plt.plot(shape[:, 0], shape[:, 1])
        j += 1

    target = utils.pload(res_dir / "target")
    target = np.append(target, [target[0, :]], axis=0)
    plt.plot(target[:, 0], target[:, 1], 'd:', label='Target')
    plt.legend(loc='best')
    plt.savefig(res_dir / 'mismatch.pdf')
    plt.clf()


def plot_theta_means(res_dir: Path):

    # count pickles
    num_t_means = 0
    prefix = "t_mean_iter="
    for fn in os.listdir(res_dir):
        if fn.startswith(prefix):
            num_t_means += 1
    print("Number of t_mean files:", num_t_means)

    # load pickles
    t_means = []
    for i in range(num_t_means):
        t_mean = utils.pload(res_dir / (prefix + str(i)))
        t_means.append(t_mean)

    # plot pickles
    print("Plotting theta means...")
    plt.grid()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    circ = lambda t: np.array([np.cos(t), np.sin(t)])
    j = 0
    while j < num_t_means:
        t_means_shape = np.array(list(map(circ, t_means[j])))
        shape = np.append(t_means_shape, [t_means_shape[0, :]], axis=0)
        plt.plot(shape[:, 0], shape[:, 1])
        j += 1

    truth_path = res_dir / "t_truth"

    if truth_path.exists():
        t_truth = utils.pload(truth_path)
        t_truth = np.array(list(map(circ, t_truth)))
        t_truth = np.append(t_truth, [t_truth[0, :]], axis=0)
        plt.plot(t_truth[:, 0], t_truth[:, 1], 'd:')

    plt.savefig(res_dir / 't_means.pdf')
    plt.clf()


def plot_alphas(res_dir: Path):
    for suffix, tex in [('momentum', '\\mathbf{P}'), ('thetas', '\\mathbf{\\Theta}')]:
        if os.path.isfile(res_dir / 'alphas'):
            print(r'Plotting $\alphas_{}$...'.format(suffix))
            alphas = utils.pload(res_dir / 'alphas')
            plt.grid()
            plt.xlabel('Iteration $n$')
            plt.ylabel(r'$\ln(\alpha_n^{})$'.format(tex))
            plt.semilogy(range(1, len(alphas) + 1), alphas)
            plt.savefig(res_dir / 'alphas_{}.pdf'.format(suffix))
            plt.clf()


def plot_landmarks(lms: np.array, label: str, linestyle: str, marker: str, path: Path):
    lms = np.append(lms, [lms[0, :]], axis=0)
    plt.figure()
    plt.plot(lms[:, 0], lms[:, 1], label=label, linestyle=linestyle, marker=marker)
    plt.legend(loc='best')
    plt.savefig(str(path))
    plt.close()


def plot_momentum(xs: np.array, ns: np.array, path: Path):
    plt.figure()
    plt.plot(xs, ns, label=reparam_label, linestyle=reparam_linestyle, marker=reparam_marker)
    plt.legend(loc='best')
    plt.savefig(str(path))
    plt.close()


def plot_initial_data(path: Path, xs: np.array, ns: np.array = None, ms: np.array = None):
    fig, ax = plt.subplots()
    lns = None
    if ns is not None:
        ln_ns = ax.plot(xs, ns, label=reparam_label, linestyle=reparam_linestyle)
        lns = ln_ns
    if ms is not None:
        from matplotlib.ticker import FormatStrFormatter
        ax2 = ax.twinx()
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3e'))
        ln_ms = ax2.plot(xs, ms, label=momentum_label, linestyle=momentum_linestyle)
        if lns is not None:
            lns += ln_ms
        else:
            lns = ln_ms

    ax.legend(lns, [lb.get_label() for lb in lns], loc='best')
    plt.savefig(str(path))
    plt.close()


if __name__ == "__main__":
    res_dir = Path(sys.argv[1])
    for p in res_dir.glob('*'):
        if not p.is_dir():
            continue
        plot_consensus(p)
        plot_errors(p)
        plot_shape_means(p)
        plot_theta_means(p)
        plot_mismatch(p)
