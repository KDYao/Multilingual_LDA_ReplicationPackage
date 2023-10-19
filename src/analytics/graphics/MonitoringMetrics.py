"""
This script plots the curve of metric values as the number of iterations increases
"""
import os
import json
import subprocess
import logging
import re
import matplotlib.pyplot as plt
import src.resources.utils.utils as ut

log = logging.getLogger(__name__)

global graph_out_dir

def grepJsonRes(f):
    """
    This function extract those training results stored in JSON format from logs
    Parameters
    ----------
    f: log file

    Returns
    -------
    """
    # Get lines with Coherence value
    cmd = 'grep -e "Coherence" {}'.format(f)
    out = subprocess.check_output(cmd, shell=True).decode('utf-8')
    iter_dict = {}
    for i, d in enumerate(out.split('\n')):
        if d == '':
            continue
        try:
            iter_dict[i] = json.loads(d.replace("\'", "\""))
        except Exception as e:
            log.error(e)
    return iter_dict

def getbestparamHistory(f_log, df_db, cost_name):
    """
    This function gets the parameters history and see how they converge
    Parameters
    ----------
    f

    Returns
    -------
    """
    # Load from db has higher priority
    tests = []
    costs = []
    if df_db is not None:
        # Sort by cost value
        total_tests = df_db.shape[0]
        temp_idx = total_tests
        # Get all turning points
        for idx, row in df_db.iterrows():
            if temp_idx > idx:
                tests.insert(0, idx)
                costs.insert(0, row['time'])
                temp_idx = idx
            if temp_idx == 0:
                break

        # Add end
        if total_tests != tests[-1]:
            tests.append(total_tests)
            costs.append(costs[-1])

        return (tests, costs)

    if os.path.isfile(f_log):
        cmd = 'grep -e "best" {}'.format(f_log)
        out = subprocess.check_output(cmd, shell=True).decode('utf-8')
        print(cmd)
        for line in out.split('\n'):
            if line == '':
                continue
            test = int(re.findall(r'tests=(\d+),', line)[0])
            cost = float(re.findall(r'cost time=(.*),', line)[0])

            tests.append(test)
            if cost_name.lower() == 'perplexity':
                costs.append(cost)
            else:
                costs.append(-1 * cost)
        return (tests, costs)
    else:
        raise FileNotFoundError('The following files are not found:\n DB: {f_db}\n LOG: {f_log}'.format(
            f_db="Not DB Input", f_log=f_log
        ))


def plotBestParamHistory(data:tuple, lda_type:str, cost_name: str, includeBaseline=True):
    """
    Ouptut cost changes by tuning iterations
    Parameters
    ----------
    data
    out_f

    Returns
    -------

    """
    # check if plot
    tests, costs = data
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(8, 6), dpi=300)

    #plt.title('[{}] Evaluation metric optimization'.format(lda_type))

    # if metric == 'PERPLEXITY':
    #     plt.plot(tests, np.log(costs), color="blue", label='optimize')
    #     plt.ylabel('log(%s)' % cost_name)
    # else:
    #     plt.plot(tests, costs, color="blue", label='optimize')
    #     plt.ylabel(cost_name)

    if metric == 'PERPLEXITY':
        plt.yscale('log')

    plt.plot(tests, costs, color="blue", label='optimize')
    plt.ylabel(cost_name)
    plt.xlabel('Optimization iterations')

    # This is hard-coded; we do not have time for flexibility concerns
    yrange_dict = {
        'COHERENCE': [0.35, 0.75],
        'AGGREGATED': [0.0, 0.9],
        'INTERCORPUSRATIO': [0.0, 1.1],
        'PERPLEXITY': [1.0, 8000.0]
    }

    axes = plt.gca()
    if cost_name in yrange_dict.keys():
        axes.set_ylim(yrange_dict[metric])

    # If we need to include baseline
    if includeBaseline:
        try:
            f_baseline = os.path.join(
                *[ut.get_project_root(), 'result', 'baseline', 'LDABaseLine_%s.csv' % lda_type]
            )
            if os.path.isfile(f_baseline):
                med_val, med_idx = ut.get_median_vals(f=f_baseline, col=cost_name)
                # if metric == 'PERPLEXITY':
                #     plt.axhline(y=np.log(med_val), color="red", linestyle="--", label='default')
                # else:
                #     plt.axhline(y=med_val, color="red", linestyle="--", label='default')
                plt.axhline(y=med_val, color="red", linestyle="--", label='default')
                plt.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15))
            else:
                print('%s Not Found' % f_baseline)
        except Exception as e:
            print('Unable to add median baseline %e' % e)
    plt.show()

    out_dir = os.path.join(graph_out_dir, 'CostHistory')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    fig.savefig(os.path.join(out_dir, 'cost_{}_{}.pdf'.format(lda_type, cost_name)))
    plt.close('all')

def plotBestParamHistory_SubPlots(best_params_list: list, lda_list:list, metric: str, includeBaseline=True):
    """
    Ouptut cost changes by tuning iterations
    All models of the same evaluaiton metric will be aggregated in a single plot
    Parameters
    ----------
    data
    out_f

    Returns
    -------

    """

    fig, subplt_list = plt.subplots(ncols=3)

    for idx, subplt in enumerate(subplt_list):
        data = best_params_list[idx]
        lda_type = lda_list[idx]
        # check if plot
        tests, costs = data
        #subplt.rcParams.update({'font.size': 18})
        subplt.plot(tests, costs, color="blue", label='optimize')
        #plt.title('[{}] Evaluation metric optimization'.format(lda_type))
        plt.xlabel('Optimization iterations')
        plt.ylabel(metric)

        # If we need to include baseline
        if includeBaseline:
            try:
                f_baseline = os.path.join(
                    *[ut.get_project_root(), 'result', 'baseline', 'LDABaseLine_%s.csv' % lda_type]
                )
                if os.path.isfile(f_baseline):
                    med_val, med_idx = ut.get_median_vals(f=f_baseline, col=metric)
                    plt.axhline(y=med_val, color="red", linestyle="--", label='default')
                    #plt.legend(frameon=False, loc='upper center', ncol=2)
            except Exception as e:
                print('Unable to add median baseline %e' % e)
    plt.show()

    out_dir = os.path.join(graph_out_dir, 'CostHistory')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    fig.savefig(os.path.join(out_dir, 'cost_{}_aggregated.pdf'.format(lda_type)))


def normalize(lst):
    """
    A simple way to normalize list of numbers
    Parameters
    ----------
    lst

    Returns
    -------

    """
    v_min, v_max = min(lst), max(lst)
    return [(x-v_min)/(v_max-v_min) for x in lst]

def plotMetricsComparison(metrics_compared, type):
    """
    Compare metrics in the same graph with normalization
    Parameters
    ----------
    metrics_compared

    Returns
    -------

    """
    fig = plt.figure(figsize=(8, 6), dpi=300)
    tests = metrics_compared.pop('Tests')

    for metric, metric_data in metrics_compared.items():
        plt.plot(tests, normalize(metric_data), label=metric)
    plt.legend()
    plt.title('[{}] Compare metric changes by tuning iterations'.format(type))
    plt.xlabel('Number of Iterations')
    plt.ylabel('Metrics (normalized)')
    plt.show()

    out_dir = os.path.join(graph_out_dir, 'CompareMetrics')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    fig.savefig(os.path.join(out_dir, 'comp_metrics_{}.pdf'.format(type)))


def getOptimalSets(f_log, df_db, lda_type, metric, isSave=True):
    """
    The last line of log saves the optimal settings of parameter sets
    You can also read it from PROJ_ROOT/HyperTuning/OptmialParam.json
    Parameters
    ----------
    f

    Returns
    -------

    """

    f_out_json = os.path.join(*[ut.get_project_root(), 'result', 'bestparam', 'BestParam.json'])

    if df_db is not None:
        # Load from db has higher priority
        best_val, best_params = ut.findBestParamSet(df_db)

        if isSave:
            if not os.path.isdir(os.path.dirname(f_out_json)):
                os.makedirs(os.path.dirname(f_out_json))
            best_param_all = {**{'BestVal': best_val}, **best_params}
            # Update OptimalParam.json
            if os.path.isfile(f_out_json):
                with open(f_out_json, 'r+') as f_update:
                    res = json.load(f_update)
                    if metric in res.keys():
                        res[metric][lda_type] = best_param_all
                    else:
                        res[metric] = {lda_type: best_param_all}
                    # Rewrite json result
                    # Seek function finds the beginning of the file
                    f_update.seek(0)
                    f_update.write(json.dumps(res, indent=4))
                    # Use truncate for inplace replacement
                    f_update.truncate()
            else:
                # Create a new json result file for that metric
                with open(f_out_json, 'w') as f_update:
                    res = {
                        metric: {
                            lda_type: best_param_all
                        }
                    }
                    f_update.write(json.dumps(res, indent=4))

        return best_params
    elif os.path.isfile(f_log):
        cmd = 'tail -n 3 {}'.format(f_log)
        out = subprocess.check_output(cmd, shell=True).decode('utf-8')
        return json.loads(re.findall(r'\{.*?\}', out)[0].replace("\'", "\""))
    else:
        raise FileNotFoundError('The following files are not found:\n DB: {f_db}\n LOG: {f_log}'.format(
            f_db="Not DB Input", f_log=f_log
        ))

def getParamsets(f):
    """
    This function extract the model training parameters which are allocated by tuner
    Parameters
    ----------
    f: log file

    Returns
    -------
    """
    # Get lines with Coherence value
    cmd = 'grep -e "num-topics" {}'.format(f)
    out = subprocess.check_output(cmd, shell=True).decode('utf-8')
    iter_params_dict = {}
    pattern = re.compile(r'--num-topics\s+(\d+)\s+--alpha\s+(\d+\.\d+)\s+--beta\s+(\d+\.\d+)')
    for i, d in enumerate(pattern.findall(out)):
        numTopics, alpha, beta = d
        iter_params_dict[i] = {
            'numTopics': int(numTopics), 'alpha': float(alpha), 'beta': float(beta)
        }
    return iter_params_dict


def compare_metrics(iter_res, best_params, metrics_to_compare):
    """
    This function evaluate the changes of different metrics as tests evloves
    Since currently we use Perplexity as cost for evaluation, and best params returns:
        By X times tests, we have the best cost (perplextiy) value
    We intend to use the same X axis ticks and see if other metrics also show such a monotonous trend
    If we observe fluctuations, that means only considering perplexity does not cover all metrics we want to evaluate

    Parameters
    ----------
    iter_res A dictionary shows the number of tests and the corresponding evaluation metrics to that test
    best_params A tuple contains the number of tests and the best cost value till that number of tests
    metrics_to_compare A list that contains the list of metrics need to be compared

    Returns
    -------

    """
    tests, costs = best_params
    next_tick = tests.pop(0)

    # This dictionary temporarily saves the best metric value till X tests
    tmp_dict = {}
    metric_func = {}
    metric_val_at_tick = {}
    metric_val_at_tick['Tests'] = list(tests)
    for metric in metrics_to_compare:
        metric_val_at_tick[metric] = []
        # Maximize coherence and log-alignment, minimize perplexity
        if metric.lower() in ['perplexity']:
            metric_func[metric] = min
            tmp_dict[metric] = float("inf")
        else:
            metric_func[metric] = max
            tmp_dict[metric] = 0.0

    for i, metrics_dict in iter_res.items():
        idx = i + 1
        for metric, func in metric_func.items():
            tmp_dict[metric] = func(float(metrics_dict[metric]), tmp_dict[metric])
            # When reach the tick
            if idx == next_tick:
                metric_val_at_tick[metric].append(tmp_dict[metric])
        if idx == next_tick:
            if tests:
                next_tick = tests.pop(0)
            else:
                return metric_val_at_tick
    return metric_val_at_tick


def plotParamsSets(iter_paramsets, type):
    """
    Compare metrics in the same graph with normalization
    Parameters
    ----------
    metrics_compared

    Returns
    -------

    """
    fig = plt.figure(figsize=(8, 6), dpi=300)
    out_dir = os.path.join(graph_out_dir, 'ParamSetsChanges')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    params = ['numTopics', 'alpha', 'beta']
    tests = iter_paramsets.keys()
    # Separate plots
    for param in params:
        plt.plot(tests, [x[param] for x in iter_paramsets.values()], label=param)
        #plt.legend()
        plt.title('[{}] Param {} changes by tuning iterations'.format(type, param))
        plt.xlabel('Number of Iterations')
        plt.ylabel(param)
        #plt.show()

        fig.savefig(os.path.join(out_dir, 'param_{}_{}.pdf'.format(param, type)))
        plt.clf()

    fig = plt.figure(figsize=(12, 8), dpi=300)
    # Compare norm
    for param in params:
        plt.plot(tests, normalize([x[param] for x in iter_paramsets.values()]), label=param)
    plt.legend()
    plt.title('[{}] Paramset changes by tuning iterations'.format(type, param))
    plt.xlabel('Number of Tests')
    plt.ylabel('Parameters (Normalized)')
    plt.show()
    # Save and clean
    fig.savefig(os.path.join(out_dir, 'paramset_normalized.pdf'))
    plt.clf()


if __name__ == '__main__':

    res_root_dir = os.path.join(ut.get_project_root(), 'out')
    res_exp_dirname = ''

    evaluate_ldas = ['LDA1', 'LDA2', 'LDA3']
    evaluate_metrics = ['PERPLEXITY', 'COHERENCE', 'AGGREGATED', 'INTERCORPUSRATIO']

    if res_exp_dirname != '':
        res_dir = os.path.join(res_root_dir, res_exp_dirname)
    else:
        res_dir = res_root_dir

    for metric in evaluate_metrics:
        all_models_best_params = []
        for lda in evaluate_ldas:
            f_log = os.path.join(*[res_dir, metric, 'log', 'output_%s.log' % lda])
            d_db = os.path.join(*[res_dir, metric, 'OpenTuner', lda, 'opentuner.db'])
            try:
                f_db = ut.list_files_by_time(dir=d_db, ext='.db')[0]
                df_db = ut.loadtable(dbpath=f_db, metric=metric)
            except Exception:
                df_db = None
            graph_out_dir = os.path.join(*[res_dir, metric, 'Graphs', lda])
            # Get opt settings
            if 'paraeval' in f_log:
                opt_settings = getOptimalSets(f_log=f_log, df_db=df_db, lda_type=lda, metric=metric, isSave=False)
            else:
                opt_settings = getOptimalSets(f_log=f_log, df_db=df_db, lda_type=lda, metric=metric, isSave=True)
            best_params = getbestparamHistory(f_log=f_log, df_db=df_db, cost_name=metric)
            plotBestParamHistory(best_params, lda, metric)
            all_models_best_params.append(best_params)

            # # Get iteration results
            # iter_res = grepJsonRes(f_log)
            # #metrics_to_compare = ['Coherence', 'Perplexity', 'AggregatedMetric']
            # metrics_to_compare = ['Coherence', 'Perplexity']
            # metrics_compared = compare_metrics(iter_res, best_params, metrics_to_compare)
            # plotMetricsComparison(metrics_compared, lda)
            #
            # # Get hyper param changing trend
            # iter_paramsets = getParamsets(f_log)
            # plotParamsSets(iter_paramsets, lda)


        #plotBestParamHistory_SubPlots(best_params_list=all_models_best_params, lda_list=evaluate_ldas, metric=metric)