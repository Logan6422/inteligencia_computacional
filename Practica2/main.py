import neurona as n
import capa as c
import red as r

input = [1, 0.5]; #bias interno
arquitectura = [2,3,2,1];
eta = 0.5;

prueba = r.red(eta,2,arquitectura);
prueba.forward_pass(input);

# #prueba salidas por capa
# for i in range(len(prueba.capas)):
#     print(prueba.capas[i].output);
#     print('\n')