from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score


class Model:
    def __init__(self, train_x, train_y, test_x, test_y, logger_object):
        self.logger = logger_object
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.multiclass_average = 'micro'
        self.param_grid_xgboost = {'max_depth': range(5, 6),
                                   'n_estimators': range(50, 60, 10),
                                   'learning_rate': [0.5, 0.1, 0.01, 0.001]}
        self.param_grid_random_forest = {'n_estimators': range(50, 150, 20),
                                         'criterion': ['gini', 'entropy'],
                                         'max_depth': range(3, 8)}
        self.param_grid_logistic_regression = {'penalty': ['l2', 'none']}
        self.param_grid_svc = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                               'degree': range(3, 10)}
        self.param_grid_decision_tree = {'criterion': ['gini', 'entropy'],
                                         'max_depth': range(3, 8)}

    def train_xgboost(self):
        try:
            self.logger.log_training('Finding Best Hyper-Parameters for XGBoost')
            model_params = self.get_best_params_xgboost()
            if model_params is not False:
                (max_depth, n_estimators, learning_rate) = model_params
                model = XGBClassifier(objective='binary:logistic',
                                      max_depth=max_depth,
                                      n_estimators = n_estimators,
                                      learning_rate=learning_rate)
                self.logger.log_training(
                    'Training XGBoost Classifier with the Following HyperParameters :: '
                    'max_depth: {}, n_estimators: {}, learning_rate: {}'.format(max_depth, n_estimators, learning_rate)
                )
                model.fit(self.train_x, self.train_y)
                return model
            else:
                raise Exception('FAILED: Error while training Model for XGBoost')
        except Exception:
            self.logger.log_training('ERROR: MODEL SELECTION: Error while training XGBoost Model')
            return False

    def train_random_forest(self):
        try:
            self.logger.log_training('Finding Best Hyper-Parameters for Random Forest')
            model_params = self.get_best_params_random_forest()
            if model_params is not False:
                (max_depth, n_estimators, criterion) = model_params
                model = RandomForestClassifier(max_depth=max_depth,
                                               n_estimators=n_estimators,
                                               criterion=criterion)
                self.logger.log_training(
                    'Training Random Forest Classifier with the Following HyperParameters ::'
                    ' max_depth: {}, n_estimators: {}, criterion: {}'.format(max_depth, n_estimators, criterion)
                )
                model.fit(self.train_x, self.train_y)
                return model
            else:
                raise Exception('Error while training the best Model for Random Forest')
        except Exception:
            self.logger.log_training('ERROR: MODEL SELECTION: Error while training Random Forest Model')
            return False

    def train_logistic_regression(self):
        try:
            self.logger.log_training('Finding Best Hyper-Parameters for Logistic Regression')
            model_params = self.get_best_params_logistic_regression()
            if model_params is not False:
                penalty = model_params
                model = LogisticRegression(penalty=penalty)
                self.logger.log_training(
                    'Training Logistic Regression Classifier with the Following HyperParameters :: penalty: {}'.format(
                        penalty
                    )
                )
                model.fit(self.train_x, self.train_y)
                return model
            else:
                raise Exception('Error while training the best Model for Logistic Regression')
        except Exception:
            self.logger.log_training('ERROR: MODEL SELECTION: Error while training Logistic Regression Model')
            return False

    def train_decision_tree(self):
        try:
            self.logger.log_training('Finding best Hyper-Parameters for Decision Tree')
            model_params = self.get_best_param_decision_tree()
            if model_params is not False:
                (max_depth, criterion) = model_params
                model = DecisionTreeClassifier(max_depth=max_depth,
                                               criterion=criterion)
                self.logger.log_training('Training Decision Tree Classifier with the Following HyperParameters ::'
                                         ' max_depth: {}, criterion: {}'.format(max_depth, criterion)
                                         )
                model.fit(self.train_x, self.train_y)
                return model
            else:
                raise Exception('Error while training the best Model for Decision Tree')
        except Exception:
            self.logger.log_training('ERROR: MODEL SELECTION: Error while training Decision Tree Model')
            return False

    def train_svc(self):
        try:
            self.logger.log_training('Finding best Hyper-Parameters for Support Vector Machine')
            model_params = self.get_best_param_svc()
            if model_params is not False:
                (kernel, degree) = model_params
                model = SVC(kernel=kernel,
                            degree=degree)
                self.logger.log_training('Training Support Vector Classifier with the Following HyperParameters ::'
                                         ' kernel: {}, degree: {}'.format(kernel, degree)
                                         )
                model.fit(self.train_x, self.train_y)
                return model
            else:
                raise Exception('Error while training the best Model for Random Forest')
        except Exception:
            self.logger.log_training('ERROR: MODEL SELECTION: Error while training Support Vector Machine Model')
            return False

    def get_best_params_xgboost(self):
        try:
            grid = GridSearchCV(XGBClassifier(objective='binary:logistic'),
                                param_grid=self.param_grid_xgboost,
                                cv=5, verbose=3)
            grid.fit(self.train_x, self.train_y)

            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']
            learning_rate = grid.best_params_['learning_rate']
            retvar = (max_depth, n_estimators, learning_rate)
            return retvar
        except Exception as e:
            self.logger.log_training('MODEL SELECTION: Error while getting best Parameters for XGBoost Classifier')
            return False

    def get_best_params_random_forest(self):
        try:
            clf = RandomForestClassifier()
            grid = GridSearchCV(clf,
                                param_grid=self.param_grid_random_forest,
                                cv=5, verbose=3)
            grid.fit(self.train_x, self.train_y)
            n_estimators = grid.best_params_['n_estimators']
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            retvar = (max_depth, n_estimators, criterion)
            return retvar
        except Exception:
            self.logger.log_training('MODEL SELECTION: Error while getting best Parameters for Random Forest Classifier')
            return False

    def get_best_params_logistic_regression(self):
        try:
            clf = LogisticRegression()
            grid = GridSearchCV(clf,
                                param_grid=self.param_grid_logistic_regression,
                                cv=5, verbose=3)
            grid.fit(self.train_x, self.train_y)
            penalty = grid.best_params_['penalty']
            return penalty
        except Exception:
            self.logger.log_training(
                'MODEL SELECTION: Error while getting best Parameters for Logistic Regression Classifier'
            )
            return False

    def get_best_param_svc(self):
        try:
            clf = SVC()
            grid = GridSearchCV(clf,
                                param_grid=self.param_grid_svc,
                                cv=5, verbose=3)
            grid.fit(self.train_x, self.train_y)
            kernel = grid.best_params_['kernel']
            degree = grid.best_params_['degree']
            retvar = (kernel, degree)
            return retvar
        except Exception as e:
            self.logger.log_training(
                'MODEL SELECTION: Error while getting best Parameters for Support Vector Classifier'
            )
            return False

    def get_best_param_decision_tree(self):
        self.param_grid_decision_tree = {'criterion': ['gini', 'entropy'],
                                         'max_depth': range(5, 15)}
        try:
            clf = DecisionTreeClassifier()
            grid = GridSearchCV(clf,
                                param_grid=self.param_grid_decision_tree,
                                cv=5, verbose=3)
            grid.fit(self.train_x, self.train_y)
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            retvar = (max_depth, criterion)
            return retvar
        except Exception:
            self.logger.log_training(
                'MODEL SELECTION: Error while getting best Parameters for Support Vector Classifier'
            )
            return False

    def get_best_model(self):
        """
        Trains all the Available models and Selects the best one based on the AUC Score
        :return:
        """
        try:
            self.logger.log_training('MODEL SELECTION: Multi Model Training -- STARTED')
            #xgboost_clf = self.train_xgboost()
            random_forest_clf = self.train_random_forest()
            #logistic_regression_clf = self.train_logistic_regression()
            decision_tree_clf = self.train_decision_tree()
            #support_vector_clf = self.train_svc()
            self.logger.log_training('MODEL SELECTION: Multi Model Training -- COMPLETED')

            models_dictionary = { #'XGBoost': xgboost_clf,
                                 'Random_Forest': random_forest_clf,
                                 'Logistic_Regression': False, #logistic_regression_clf,
                                 'Decision_Tree': decision_tree_clf,
                                 'Support_Vector_Machine': False #support_vector_clf
            }
            scores_dictionary = {'XGBoost': None,
                                 'Random_Forest': None,
                                 'Logistic_Regression': None,
                                 'Decision_Tree': None,
                                 'Support_Vector Machine': None}

            select_by_accuracy = False

            self.logger.log_training('MODEL SELECTION: Making Predictions and Storing Scores -- STARTED')
            for model_name in models_dictionary:
                model = models_dictionary[model_name]
                if model is not False:  # Checking if model is trained properly or not
                    prediction = model.predict(self.test_x)
                    try:
                        scores_dictionary[model_name] = roc_auc_score(self.test_y, prediction)
                    except Exception as e:
                        self.logger.log_training(
                            'MODEL SELECTION : FAILED to select by ROC-AUC Score, Using Accuracy Score instead'
                        )
                        select_by_accuracy = True

            if select_by_accuracy is True:
                for model_name in models_dictionary:
                    model = models_dictionary[model_name]
                    if model is not False:  # Checking if model is trained properly or not
                        prediction = model.predict(self.test_x)
                        scores_dictionary[model_name] = accuracy_score(self.test_y, prediction)

            self.logger.log_training('MODEL SELECTION: Making Predictions and Storing Scores -- COMPLETED')

            self.logger.log_training('MODEL SELECTION: Comparing and Finding Best Model')

            # Selecting Valid Models
            final_scores_dict = {}
            for model_name in scores_dictionary:
                if scores_dictionary[model_name] is not None:
                    final_scores_dict[model_name] = scores_dictionary[model_name]

            # self.logger.log_training('Trained Following Models with given AUC Scores: {}'.format(final_scores_dict))

            best_model_name = max(final_scores_dict, key=lambda x: final_scores_dict[x])
            best_model = models_dictionary[best_model_name]
            self.logger.log_training('MODEL SELECTION: COMPLETED: selected -- {}'.format(best_model_name))
            best_model_metrics = self.metrics(best_model, best_model_name)
            retvar = (best_model, best_model_name, best_model_metrics)
            return retvar

        except Exception as e:
            self.logger.log_training('CRITICAL: * Training Failed * please check the previous log for reason')
            raise e

    def metrics(self, best_model, best_model_name):
        self.logger.log_training("Preparing Metrics to be stored and reviewed")
        predictions = best_model.predict(self.test_x)
        accuracy = accuracy_score(self.test_y, predictions)
        recall = recall_score(self.test_y, predictions, average=self.multiclass_average)
        precision = precision_score(self.test_y, predictions, average=self.multiclass_average)
        f1 = f1_score(self.test_y, predictions, average=self.multiclass_average)
        metrics_dict = {'model': str(best_model_name),
                        'accuracy': str(accuracy),
                        'recall': str(recall),
                        'precision': str(precision),
                        'f1': str(f1)}
        return metrics_dict
