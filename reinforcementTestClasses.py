# reinforcementTestClasses.py
# ---------------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import testClasses
import random, math, traceback, sys, os
import layout, textDisplay, pacman, gridworld
import time
from util import Counter, TimeoutFunction, FixedRandom
from collections import defaultdict
from pprint import PrettyPrinter
from hashlib import sha1
from functools import reduce
pp = PrettyPrinter()
VERBOSE = False

LIVINGREWARD = -0.1
NOISE = 0.2

class ApproximateQLearningTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(ApproximateQLearningTest, self).__init__(question, testDict)
        self.discount = float(testDict['discount'])
        self.grid = gridworld.Gridworld(parseGrid(testDict['grid']))
        if 'noise' in testDict: self.grid.setNoise(float(testDict['noise']))
        if 'livingReward' in testDict: self.grid.setLivingReward(float(testDict['livingReward']))
        self.grid = gridworld.Gridworld(parseGrid(testDict['grid']))
        self.env = gridworld.GridworldEnvironment(self.grid)
        self.epsilon = float(testDict['epsilon'])
        self.learningRate = float(testDict['learningRate'])
        self.extractor = 'IdentityExtractor'
        if 'extractor' in testDict:
            self.extractor = testDict['extractor']
        self.opts = {'actionFn': self.env.getPossibleActions, 'epsilon': self.epsilon, 'gamma': self.discount, 'alpha': self.learningRate}
        numExperiences = int(testDict['numExperiences'])
        maxPreExperiences = 10
        self.numsExperiencesForDisplay = list(range(min(numExperiences, maxPreExperiences)))
        self.testOutFile = testDict['test_out_file']
        if maxPreExperiences < numExperiences:
            self.numsExperiencesForDisplay.append(numExperiences)

    def writeFailureFile(self, string):
        with open(self.testOutFile, 'w') as handle:
            handle.write(string)

    def removeFailureFileIfExists(self):
        if os.path.exists(self.testOutFile):
            os.remove(self.testOutFile)

    def execute(self, grades, moduleDict, solutionDict):
        failureOutputFileString = ''
        failureOutputStdString = ''
        for n in self.numsExperiencesForDisplay:
            testPass, stdOutString, fileOutString = self.executeNExperiences(grades, moduleDict, solutionDict, n)
            failureOutputStdString += stdOutString
            failureOutputFileString += fileOutString
            if not testPass:
                self.addMessage(failureOutputStdString)
                self.addMessage('For more details to help you debug, see test output file %s\n\n' % self.testOutFile)
                self.writeFailureFile(failureOutputFileString)
                return self.testFail(grades)
        self.removeFailureFileIfExists()
        return self.testPass(grades)

    def executeNExperiences(self, grades, moduleDict, solutionDict, n):
        testPass = True
        qValuesPretty, weights, actions, lastExperience = self.runAgent(moduleDict, n)
        stdOutString = ''
        fileOutString = "==================== Iteration %d ====================\n" % n
        if lastExperience is not None:
            fileOutString += "Agent observed the transition (startState = %s, action = %s, endState = %s, reward = %f)\n\n" % lastExperience
        weightsKey = 'weights_k_%d' % n
        if weights == eval(solutionDict[weightsKey]):
            fileOutString += "Weights at iteration %d are correct." % n
            fileOutString += "   Student/correct solution:\n\n%s\n\n" % pp.pformat(weights)
        for action in actions:
            qValuesKey = 'q_values_k_%d_action_%s' % (n, action)
            qValues = qValuesPretty[action]
            if self.comparePrettyValues(qValues, solutionDict[qValuesKey]):
                fileOutString += "Q-Values at iteration %d for action '%s' are correct." % (n, action)
                fileOutString += "   Student/correct solution:\n\t%s" % self.prettyValueSolutionString(qValuesKey, qValues)
            else:
                testPass = False
                outString = "Q-Values at iteration %d for action '%s' are NOT correct." % (n, action)
                outString += "   Student solution:\n\t%s" % self.prettyValueSolutionString(qValuesKey, qValues)
                outString += "   Correct solution:\n\t%s" % self.prettyValueSolutionString(qValuesKey, solutionDict[qValuesKey])
                stdOutString += outString
                fileOutString += outString
        return testPass, stdOutString, fileOutString

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            for n in self.numsExperiencesForDisplay:
                qValuesPretty, weights, actions, _ = self.runAgent(moduleDict, n)
                handle.write(self.prettyValueSolutionString('weights_k_%d' % n, pp.pformat(weights)))
                for action in actions:
                    handle.write(self.prettyValueSolutionString('q_values_k_%d_action_%s' % (n, action), qValuesPretty[action]))
        return True

    def runAgent(self, moduleDict, numExperiences):
        agent = moduleDict['qlearningAgents'].ApproximateQAgent(extractor=self.extractor, **self.opts)
        states = [state for state in self.grid.getStates() if len(self.grid.getPossibleActions(state)) > 0]
        states.sort()
        randObj = FixedRandom().random
        # choose a random start state and a random possible action from that state
        # get the next state and reward from the transition function
        lastExperience = None
        for i in range(numExperiences):
            startState = randObj.choice(states)
            action = randObj.choice(self.grid.getPossibleActions(startState))
            (endState, reward) = self.env.getRandomNextState(startState, action, randObj=randObj)
            lastExperience = (startState, action, endState, reward)
            agent.update(*lastExperience)
        actions = list(reduce(lambda a, b: set(a).union(b), [self.grid.getPossibleActions(state) for state in states]))
        qValues = {}
        weights = agent.getWeights()
        for state in states:
            possibleActions = self.grid.getPossibleActions(state)
            for action in actions:
                if action not in qValues:
                    qValues[action] = {}
                if action in possibleActions:
                    qValues[action][state] = agent.getQValue(state, action)
                else:
                    qValues[action][state] = None
        qValuesPretty = {}
        for action in actions:
            qValuesPretty[action] = self.prettyValues(qValues[action])
        return (qValuesPretty, weights, actions, lastExperience)

    def prettyPrint(self, elements, formatString):
        pretty = ''
        states = self.grid.getStates()
        for ybar in range(self.grid.grid.height):
            y = self.grid.grid.height-1-ybar
            row = []
            for x in range(self.grid.grid.width):
                if (x, y) in states:
                    value = elements[(x, y)]
                    if value is None:
                        row.append('   illegal')
                    else:
                        row.append(formatString.format(elements[(x,y)]))
                else:
                    row.append('_' * 10)
            pretty += '        %s\n' % ("   ".join(row), )
        pretty += '\n'
        return pretty

    def prettyValues(self, values):
        return self.prettyPrint(values, '{0:10.4f}')

    def prettyPolicy(self, policy):
        return self.prettyPrint(policy, '{0:10s}')

    def prettyValueSolutionString(self, name, pretty):
        return '%s: """\n%s\n"""\n\n' % (name, pretty.rstrip())

    def comparePrettyValues(self, aPretty, bPretty, tolerance=0.01):
        aList = self.parsePrettyValues(aPretty)
        bList = self.parsePrettyValues(bPretty)
        if len(aList) != len(bList):
            return False
        for a, b in zip(aList, bList):
            try:
                aNum = float(a)
                bNum = float(b)
                # error = abs((aNum - bNum) / ((aNum + bNum) / 2.0))
                error = abs(aNum - bNum)
                if error > tolerance:
                    return False
            except ValueError:
                if a.strip() != b.strip():
                    return False
        return True

    def parsePrettyValues(self, pretty):
        values = pretty.split()
        return values


class QLearningTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(QLearningTest, self).__init__(question, testDict)
        self.discount = float(testDict['discount'])
        self.grid = gridworld.Gridworld(parseGrid(testDict['grid']))
        if 'noise' in testDict: self.grid.setNoise(float(testDict['noise']))
        if 'livingReward' in testDict: self.grid.setLivingReward(float(testDict['livingReward']))
        self.grid = gridworld.Gridworld(parseGrid(testDict['grid']))
        self.env = gridworld.GridworldEnvironment(self.grid)
        self.epsilon = float(testDict['epsilon'])
        self.learningRate = float(testDict['learningRate'])
        self.opts = {'actionFn': self.env.getPossibleActions, 'epsilon': self.epsilon, 'gamma': self.discount, 'alpha': self.learningRate}
        numExperiences = int(testDict['numExperiences'])
        maxPreExperiences = 10
        self.numsExperiencesForDisplay = list(range(min(numExperiences, maxPreExperiences)))
        self.testOutFile = testDict['test_out_file']
        if maxPreExperiences < numExperiences:
            self.numsExperiencesForDisplay.append(numExperiences)

    def writeFailureFile(self, string):
        with open(self.testOutFile, 'w') as handle:
            handle.write(string)

    def removeFailureFileIfExists(self):
        if os.path.exists(self.testOutFile):
            os.remove(self.testOutFile)

    def execute(self, grades, moduleDict, solutionDict):
        failureOutputFileString = ''
        failureOutputStdString = ''
        for n in self.numsExperiencesForDisplay:
            checkValuesAndPolicy = (n == self.numsExperiencesForDisplay[-1])
            testPass, stdOutString, fileOutString = self.executeNExperiences(grades, moduleDict, solutionDict, n, checkValuesAndPolicy)
            failureOutputStdString += stdOutString
            failureOutputFileString += fileOutString
            if not testPass:
                self.addMessage(failureOutputStdString)
                self.addMessage('For more details to help you debug, see test output file %s\n\n' % self.testOutFile)
                self.writeFailureFile(failureOutputFileString)
                return self.testFail(grades)
        self.removeFailureFileIfExists()
        return self.testPass(grades)

    def executeNExperiences(self, grades, moduleDict, solutionDict, n, checkValuesAndPolicy):
        testPass = True
        valuesPretty, qValuesPretty, actions, policyPretty, lastExperience = self.runAgent(moduleDict, n)
        stdOutString = ''
        fileOutString = "==================== Iteration %d ====================\n" % n
        if lastExperience is not None:
            fileOutString += "Agent observed the transition (startState = %s, action = %s, endState = %s, reward = %f)\n\n\n" % lastExperience
        for action in actions:
            qValuesKey = 'q_values_k_%d_action_%s' % (n, action)
            qValues = qValuesPretty[action]
            if self.comparePrettyValues(qValues, solutionDict[qValuesKey]):
                fileOutString += "Q-Values at iteration %d for action '%s' are correct." % (n, action)
                fileOutString += "   Student/correct solution:\n\t%s" % self.prettyValueSolutionString(qValuesKey, qValues)
            else:
                testPass = False
                outString = "Q-Values at iteration %d for action '%s' are NOT correct." % (n, action)
                outString += "   Student solution:\n\t%s" % self.prettyValueSolutionString(qValuesKey, qValues)
                outString += "   Correct solution:\n\t%s" % self.prettyValueSolutionString(qValuesKey, solutionDict[qValuesKey])
                stdOutString += outString
                fileOutString += outString
        if checkValuesAndPolicy:
            if not self.comparePrettyValues(valuesPretty, solutionDict['values']):
                testPass = False
                outString = "Values are NOT correct."
                outString += "   Student solution:\n\t%s" % self.prettyValueSolutionString('values', valuesPretty)
                outString += "   Correct solution:\n\t%s" % self.prettyValueSolutionString('values', solutionDict['values'])
                stdOutString += outString
                fileOutString += outString
            if not self.comparePrettyValues(policyPretty, solutionDict['policy']):
                testPass = False
                outString = "Policy is NOT correct."
                outString += "   Student solution:\n\t%s" % self.prettyValueSolutionString('policy', policyPretty)
                outString += "   Correct solution:\n\t%s" % self.prettyValueSolutionString('policy', solutionDict['policy'])
                stdOutString += outString
                fileOutString += outString
        return testPass, stdOutString, fileOutString

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            valuesPretty = ''
            policyPretty = ''
            for n in self.numsExperiencesForDisplay:
                valuesPretty, qValuesPretty, actions, policyPretty, _ = self.runAgent(moduleDict, n)
                for action in actions:
                    handle.write(self.prettyValueSolutionString('q_values_k_%d_action_%s' % (n, action), qValuesPretty[action]))
            handle.write(self.prettyValueSolutionString('values', valuesPretty))
            handle.write(self.prettyValueSolutionString('policy', policyPretty))
        return True

    def runAgent(self, moduleDict, numExperiences):
        agent = moduleDict['qlearningAgents'].QLearningAgent(**self.opts)
        states = [state for state in self.grid.getStates() if len(self.grid.getPossibleActions(state)) > 0]
        states.sort()
        randObj = FixedRandom().random
        # choose a random start state and a random possible action from that state
        # get the next state and reward from the transition function
        lastExperience = None
        for i in range(numExperiences):
            startState = randObj.choice(states)
            action = randObj.choice(self.grid.getPossibleActions(startState))
            (endState, reward) = self.env.getRandomNextState(startState, action, randObj=randObj)
            lastExperience = (startState, action, endState, reward)
            agent.update(*lastExperience)
        actions = list(reduce(lambda a, b: set(a).union(b), [self.grid.getPossibleActions(state) for state in states]))
        values = {}
        qValues = {}
        policy = {}
        for state in states:
            values[state] = agent.computeValueFromQValues(state)
            policy[state] = agent.computeActionFromQValues(state)
            possibleActions = self.grid.getPossibleActions(state)
            for action in actions:
                if action not in qValues:
                    qValues[action] = {}
                if action in possibleActions:
                    qValues[action][state] = agent.getQValue(state, action)
                else:
                    qValues[action][state] = None
        valuesPretty = self.prettyValues(values)
        policyPretty = self.prettyPolicy(policy)
        qValuesPretty = {}
        for action in actions:
            qValuesPretty[action] = self.prettyValues(qValues[action])
        return (valuesPretty, qValuesPretty, actions, policyPretty, lastExperience)

    def prettyPrint(self, elements, formatString):
        pretty = ''
        states = self.grid.getStates()
        for ybar in range(self.grid.grid.height):
            y = self.grid.grid.height-1-ybar
            row = []
            for x in range(self.grid.grid.width):
                if (x, y) in states:
                    value = elements[(x, y)]
                    if value is None:
                        row.append('   illegal')
                    else:
                        row.append(formatString.format(elements[(x,y)]))
                else:
                    row.append('_' * 10)
            pretty += '        %s\n' % ("   ".join(row), )
        pretty += '\n'
        return pretty

    def prettyValues(self, values):
        return self.prettyPrint(values, '{0:10.4f}')

    def prettyPolicy(self, policy):
        return self.prettyPrint(policy, '{0:10s}')

    def prettyValueSolutionString(self, name, pretty):
        return '%s: """\n%s\n"""\n\n' % (name, pretty.rstrip())

    def comparePrettyValues(self, aPretty, bPretty, tolerance=0.01):
        aList = self.parsePrettyValues(aPretty)
        bList = self.parsePrettyValues(bPretty)
        if len(aList) != len(bList):
            return False
        for a, b in zip(aList, bList):
            try:
                aNum = float(a)
                bNum = float(b)
                # error = abs((aNum - bNum) / ((aNum + bNum) / 2.0))
                error = abs(aNum - bNum)
                if error > tolerance:
                    return False
            except ValueError:
                if a.strip() != b.strip():
                    return False
        return True

    def parsePrettyValues(self, pretty):
        values = pretty.split()
        return values

### q7/q8
### =====
## Average wins of a pacman agent

class EvalAgentTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(EvalAgentTest, self).__init__(question, testDict)
        self.pacmanParams = testDict['pacmanParams']

        self.scoreMinimum = int(testDict['scoreMinimum']) if 'scoreMinimum' in testDict else None
        self.nonTimeoutMinimum = int(testDict['nonTimeoutMinimum']) if 'nonTimeoutMinimum' in testDict else None
        self.winsMinimum = int(testDict['winsMinimum']) if 'winsMinimum' in testDict else None

        self.scoreThresholds = [int(s) for s in testDict.get('scoreThresholds','').split()]
        self.nonTimeoutThresholds = [int(s) for s in testDict.get('nonTimeoutThresholds','').split()]
        self.winsThresholds = [int(s) for s in testDict.get('winsThresholds','').split()]

        self.maxPoints = sum([len(t) for t in [self.scoreThresholds, self.nonTimeoutThresholds, self.winsThresholds]])


    def execute(self, grades, moduleDict, solutionDict):
        self.addMessage('Grading agent using command:  python pacman.py %s'% (self.pacmanParams,))

        startTime = time.time()
        games = pacman.runGames(** pacman.readCommand(self.pacmanParams.split(' ')))
        totalTime = time.time() - startTime
        numGames = len(games)

        stats = {'time': totalTime, 'wins': [g.state.isWin() for g in games].count(True),
                 'games': games, 'scores': [g.state.getScore() for g in games],
                 'timeouts': [g.agentTimeout for g in games].count(True), 'crashes': [g.agentCrashed for g in games].count(True)}

        averageScore = sum(stats['scores']) / float(len(stats['scores']))
        nonTimeouts = numGames - stats['timeouts']
        wins = stats['wins']

        def gradeThreshold(value, minimum, thresholds, name):
            points = 0
            passed = (minimum == None) or (value >= minimum)
            if passed:
                for t in thresholds:
                    if value >= t:
                        points += 1
            return (passed, points, value, minimum, thresholds, name)

        results = [gradeThreshold(averageScore, self.scoreMinimum, self.scoreThresholds, "average score"),
                   gradeThreshold(nonTimeouts, self.nonTimeoutMinimum, self.nonTimeoutThresholds, "games not timed out"),
                   gradeThreshold(wins, self.winsMinimum, self.winsThresholds, "wins")]

        totalPoints = 0
        for passed, points, value, minimum, thresholds, name in results:
            if minimum == None and len(thresholds)==0:
                continue

            # print passed, points, value, minimum, thresholds, name
            totalPoints += points
            if not passed:
                assert points == 0
                self.addMessage("%s %s (fail: below minimum value %s)" % (value, name, minimum))
            else:
                self.addMessage("%s %s (%s of %s points)" % (value, name, points, len(thresholds)))

            if minimum != None:
                self.addMessage("    Grading scheme:")
                self.addMessage("     < %s:  fail" % (minimum,))
                if len(thresholds)==0 or minimum != thresholds[0]:
                    self.addMessage("    >= %s:  0 points" % (minimum,))
                for idx, threshold in enumerate(thresholds):
                    self.addMessage("    >= %s:  %s points" % (threshold, idx+1))
            elif len(thresholds) > 0:
                self.addMessage("    Grading scheme:")
                self.addMessage("     < %s:  0 points" % (thresholds[0],))
                for idx, threshold in enumerate(thresholds):
                    self.addMessage("    >= %s:  %s points" % (threshold, idx+1))

        if any([not passed for passed, _, _, _, _, _ in results]):
            totalPoints = 0

        return self.testPartial(grades, totalPoints, self.maxPoints)

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# File intentionally blank.\n')
        return True

def parseGrid(string):
    grid = [[entry.strip() for entry in line.split()] for line in string.split('\n')]
    for row in grid:
        for x, col in enumerate(row):
            try:
                col = int(col)
            except:
                pass
            if col == "_":
                col = ' '
            row[x] = col
    return gridworld.makeGrid(grid)