# Created by nfreundl at 19/11/2018
Feature: ParseModel
  # Enter feature description here
  # Read input, create dataframes and matrices
  # Check examples here: https://opensource.com/article/18/5/behavior-driven-python

  @Sanity
  Scenario: FormatFeature
    # Enter steps here
#    Given the basket is empty
#    When "4" cucumbers are added to the basket
#    And "6" more cucumbers are added to the basket
#    But "3" cucumbers are removed from the basket
#    Then the basket contains "7" cucumbers
    Given the feature list is empty
    When a feature list is added
    Then a dataframe is created