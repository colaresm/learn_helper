from basic_libs import *

alpha = 0.05


def normalityTest(accs_dict):
    shapiro(accs_dict["MLP"]), shapiro(accs_dict["SVM (RBF)"]), shapiro(accs_dict["SVM (Linear)"])

def anova(accs_dict):
    _, p =f_oneway(accs_dict["MLP"]), shapiro(accs_dict["SVM (RBF)"]), shapiro(accs_dict["SVM (Linear)"])

    if p <= alpha:
        print('Hipótese nula rejeitada. Dados são diferentes')
    else:
        print('Hipótese alternativa rejeitada. Resultados são iguais')

def tukey(accs_dict):
  results_clfs=pd.DataFrame(accs_dict)
  df = pd.DataFrame([(key, value) for key, values in accs_dict.items() for value in values], columns=['classifier', 'acc'])
  comp_data = MultiComparison(df['acc'], df['classifier'])

  print(comp_data.tukeyhsd())

  print(results_clfs.mean())

