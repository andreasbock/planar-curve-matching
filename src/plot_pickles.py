import matplotlib.pyplot as plt
import numpy as np
import src.utils as utils
import sys
import os

from pathlib import Path


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

shape_marker = 'x'
shape_linestyle = 'dotted'
shape_label = 'Reconstruction'

target_marker = 'd'
target_linestyle = '-.'
target_label = 'Target'

momentum_marker = 'x'
momentum_linestyle = 'dotted'
momentum_label = r'$\mathbf{p}$'


def plot_consensus(res_dir: Path):
    vnames = {'momentum': 'Consensus (momentum)'}

    def _plot_consensus(var):
        if os.path.isfile(res_dir / 'consensuses_{}'.format(var)):
            consensuses = utils.pload(res_dir / 'consensuses_{}'.format(var))
            print(f"Plotting {var} consensuses...")
            ax = plt.subplot(111)
            ax.set_xlim(left=1, right=len(consensuses))
            plt.grid()
            plt.xlabel('Iteration')
            plt.ylabel(vnames[var])
            plt.plot(range(1, len(consensuses) + 1), consensuses)
            plt.savefig(res_dir / 'consensuses_{}.pdf'.format(var))
            plt.close()
            return consensuses

    cm = _plot_consensus('momentum')

    return cm


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
        #plt.axhline(y=eta, linestyle=':', color='red', label=r'Noise level')
        plt.legend(loc='best')
        plt.savefig(res_dir / 'data_misfit.pdf')
        plt.clf()

    errors_path_momentum = res_dir / 'relative_error_momentum'
    print(r'Plotting relative error...')
    plt.figure()
    errors_momentum = utils.pload(errors_path_momentum)
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel(r'Relative error')
    plt.semilogy(range(1, len(errors_momentum) + 1), errors_momentum, label='Momentum')
    plt.legend(loc='best')
    plt.savefig(res_dir / 'relative_error.pdf')
    plt.close()

    return errors_momentum, errors


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

    #target = utils.pload(res_dir / "target")
    #target = np.append(target, [target[0, :]], axis=0)

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

    return shapes[-1], target


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
    plt.close()


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
    plt.close()


def plot_alphas(res_dir: Path):
    if os.path.isfile(res_dir / 'alphas'):
        print(r'Plotting $\alphas$...')
        alphas = utils.pload(res_dir / 'alphas')
        plt.grid()
        plt.xlabel('Iteration $n$')
        plt.ylabel(r'$\ln(\alpha_n)$')
        plt.semilogy(range(1, len(alphas) + 1), alphas)
        plt.savefig(res_dir / 'alphas.pdf')
        plt.close()


def plot_landmarks(lms: np.array, label: str, linestyle: str, marker: str, path: Path):
    lms = np.append(lms, [lms[0, :]], axis=0)
    plt.figure()
    plt.plot(lms[:, 0], lms[:, 1], label=label, linestyle=linestyle, marker=marker)
    plt.legend(loc='best')
    plt.savefig(str(path))
    plt.close()


def plot_momentum(xs: np.array, ns: np.array, path: Path):
    plt.figure()
    plt.plot(xs, ns, label=momentum_label, linestyle=momentum_linestyle, marker=momentum_marker)
    plt.legend(loc='best')
    plt.savefig(str(path))
    plt.close()


def plot_initial_data(path: Path, xs: np.array, ms: np.array = None):
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\theta$')
    lns = None
    if ms is not None:
        from matplotlib.ticker import FormatStrFormatter
        ax2 = ax.twinx()
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3e'))
        ln_ms = ax2.plot(xs, ms, label=momentum_label, linestyle=momentum_linestyle)
        if lns is not None:
            lns += ln_ms
        else:
            lns = ln_ms

    plt.savefig(str(path))
    plt.close()


if __name__ == "__main__":
    res_dir = Path(sys.argv[1])

    # iterate through solutions
    for pp in res_dir.glob('*'):
        if not pp.is_dir():
            continue

        shapes = []
        err_moms = []
        misfits = []
        cons_moms = []

        # iterate through realisations of the solutions
        ps = list(pp.glob('*'))
        for p in ps:
            if not p.is_dir():
                continue
            #if (p / "shape_means.pdf").exists():
            #    print(f"Skipping {p}, already plotted these.")
            #    continue

            cm = plot_consensus(p)
            cons_moms.append(cm)

            mf, misfit = plot_errors(p)
            misfits.append(misfit)
            err_moms.append(mf)

            #shape, target = plot_shape_means(p)
            #shapes.append(shape)

            plot_alphas(p)

        # Plot aggregation over realisations
        if len(ps) > 0:
            fig_consensus_momentum = plt.figure()
            ax_cons_mom = fig_consensus_momentum.add_subplot(111)
            ax_cons_mom.grid()
            for cm in cons_moms:
                ax_cons_mom.plot(range(1, len(cm) + 1), cm)
            ax_cons_mom.set_xlabel('Iteration')
            ax_cons_mom.set_ylabel('Consensus (momentum)')
            ax_cons_mom.set_xlim(left=1, right=len(cm))
            fig_consensus_momentum.savefig(pp / 'consensuses_momentum_avg.pdf')
            fig_consensus_momentum.clf()

            plt.clf()
            plt.close()

            # set up figures for averages
            fig_err_mom = plt.figure()
            ax_err_mom = fig_err_mom.add_subplot(111)
            ax_err_mom.grid()
            for mf in err_moms:
                ax_err_mom.plot(range(1, len(mf) + 1), mf)
            ax_err_mom.set_xlabel('Iteration')
            ax_err_mom.set_ylabel('Relative error (momentum)')
            ax_err_mom.set_xlim(left=1, right=len(mf))
            fig_err_mom.savefig(pp / 'relative-error_momentum_avg.pdf')
            fig_err_mom.clf()
            plt.clf()
            plt.close()


            fig_err_misfit = plt.figure()
            ax_err_misfit = fig_err_misfit.add_subplot(111)
            for mf in misfits:
                ax_err_misfit.semilogy(range(1, len(mf) + 1), mf)
            ax_err_misfit.grid()
            ax_err_misfit.set_xlabel('Iteration')
            ax_err_misfit.set_ylabel('Data misfit')
            ax_err_mom.set_xlim(left=1, right=len(mf))
            fig_err_misfit.savefig(pp / 'data_misfits.pdf')
            fig_err_misfit.clf()
            plt.close()

            if False:
                plt.figure()
                for shape in shapes:
                    plt.plot(shape[:, 0], shape[:, 1], linewidth=0.3)
                plt.plot(target[:, 0], target[:, 1], marker=target_marker, linestyle=target_linestyle, label=target_label)
                plt.grid()
                plt.xlabel('$x$')
                plt.ylabel('$y$')
                plt.legend(loc='best')
                plt.savefig(pp / 'shapes_avg.pdf')
                plt.clf()
                plt.close()


