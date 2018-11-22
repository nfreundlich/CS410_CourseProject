from behave import *

import pandas as pd
from feature_mining import ParseModel

use_step_matcher("re")


@given("the feature list is empty")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    print("Step1")
    context.pm = ParseModel()

    # raise NotImplementedError(u'STEP: Given the feature list is empty')


@when("a feature list is added")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    print("Step2")
    context.feature_list = context.pm.format_feature_list(feature_list=
                                          ["sound",
                                           "battery",
                                           "screen",
                                           "display"]
                                          )

    df = pd.DataFrame([["sound", 0, 0],
                       ["battery", 1, 1],
                       ["screen", 2, 2],
                       ["display", 3, 3]], columns=["feature", "feature_id", "feature_term_id"])

    assert(pd.DataFrame.equals(df, context.feature_list))
    # raise NotImplementedError(u'STEP: When a feature list is added')


@then("a dataframe is created")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    print("Step3")
    print(context.feature_list)
    # raise NotImplementedError(u'STEP: Then a dataframe is created')

