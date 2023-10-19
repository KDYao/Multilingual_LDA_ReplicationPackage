"""
Tune LDA models with OpenTuner
"""
import opentuner
import os
import sys
import json
import statistics
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter
from opentuner import FloatParameter
from opentuner import MeasurementInterface
from opentuner import Result
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.resources.utils.utils import parse_args_tune_lda, getPath, getWorkers, EvaluationMetrics
from src.LDA.RunLDAOpts import OptionLDARunner
from src.LDA.LDAOptions import LDAOptions

mallet_path = getPath('mallet')
os.environ.update({
    'MALLET_HOME': os.path.dirname(os.path.dirname(mallet_path))
})

root = getPath('root')
#TODO: FIXTHIS: allow save data by single section or by multisections
f_corpus_name = getPath('CORPUS_NAME')
f_data_dict = os.path.join(*[root, 'post-processed-data', f_corpus_name])
ldarunner = OptionLDARunner(root=root, f_data_dict=f_data_dict, mallet_path=mallet_path, workers=getWorkers())

class LDATuner(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        manipulator = ConfigurationManipulator()

        numTopics_min, numTopics_max = [int(x) for x in self.args.numTopicsRange.split('-')]
        alpha_min, alpha_max = [float(x) for x in self.args.alphaRange.split('-')]
        beta_min, beta_max = [float(x) for x in self.args.betaRange.split('-')]

        manipulator.add_parameter(
            IntegerParameter('numTopics', numTopics_min, numTopics_max))
        manipulator.add_parameter(
            FloatParameter('alpha', alpha_min, alpha_max)
        )
        manipulator.add_parameter(
            FloatParameter('beta', beta_min, beta_max)
        )
        return manipulator

    def getVal(self, res, metric_val):
        """
        Get value for param tuning.
        This is reserve for handling different formatted (dict and list) result from different configurations
        - Normal tuning returns single dictionary that include metrics such as coherence, perplexity, etc.
          Then according to the provided keyword, select the specific metric for tuning
        - Another case is the aggregation of running from 10 runs, and we use the median value for tuning
          The reason is to avoid noise and randomness in LDA training (sampling could cause variance in modeling)
          The result is a list of dictionaries, we will extract the median value of selected metric

        Parameters
        ----------
        res
        metric

        Return val for tuning
        -------

        """
        metric = EvaluationMetrics(metric_val)
        if isinstance(res, dict):
            # Desired to maximize all metrics except perplexity
            if metric == EvaluationMetrics.PERPLEXITY:
                res_cost = float(res[metric.name])
            else:
                res_cost = -1 * float(res[metric.name])
        elif isinstance(res, list):
            # Get the list of values from 10 parallel runs
            if metric == EvaluationMetrics.PERPLEXITY:
                res_list = [float(x[metric]) for x in res]
            else:
                res_list = [-1 * float(x[metric.name]) for x in res]
            res_cost = statistics.median(res_list)
        return res_cost


    def run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        cfg = desired_result.configuration.data

        lda_type = self.args.LDAType
        args = self.args
        args.numTopics = cfg['numTopics']
        args.alpha = cfg['alpha']
        args.beta = cfg['beta']
        if lda_type in [x.name for x in LDAOptions]:
            res = ldarunner.run(LDAOptions[lda_type], args)
        else:
            raise TypeError('Type %s not defined.' % lda_type)
        res_cost = self.getVal(res=res, metric_val=args.metric)
        # Return time to minimize the result
        return Result(time=res_cost)

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print("Optimal block size written to %s_result.json:" % self.args.LDAType, configuration.data)
        # self.manipulator().save_to_file(configuration.data,
        #                                 '%s_result.json' % self.args.LDAType)
        with open('%s_result.json' % self.args.LDAType, 'w') as f_out:
            f_out.write(json.dumps(configuration.data))

        if hasattr(self.args, 'metric'):
            metric_name = EvaluationMetrics(args.metric).name
            f_opt_hyper_param = os.path.join(os.path.dirname(__file__), 'OptimalParam_{}.json'.format(metric_name))
        else:
            f_opt_hyper_param = os.path.join(os.path.dirname(__file__), 'OptimalParam.json')
        # Update OptimalParam.json
        if os.path.isfile(f_opt_hyper_param):
            with open(f_opt_hyper_param, 'r+') as f_update:
                res = json.load(f_update)
                res[self.args.LDAType] = configuration.data
                # Rewrite json result
                # Seek function finds the beginning of the file
                f_update.seek(0)
                f_update.write(json.dumps(res, indent=4))
                # Use truncate for inplace replacement
                f_update.truncate()
        else:
            # Create a new json result file for that metric
            with open(f_opt_hyper_param, 'w') as f_update:
                res = {self.args.LDAType: configuration.data}
                f_update.write(json.dumps(res, indent=4))


if __name__ == '__main__':
    args, _ = parse_args_tune_lda(parents=opentuner.argparsers())
    LDATuner.main(args)
