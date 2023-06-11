### Modelo básico para classificar nomes entre ‘masculino’ e ‘feminino’

Utilizando datasets de nomes do repositório: [genderComputer](https://github.com/tue-mdse/genderComputer/tree/master/nameLists)

Consultar arquivo “model.py”.

Modelo treinado já fornecido (**model.pkl**)

Para utilizar em código externo, será necessário definir a classe **FeatureExtractor**, e o método **extract\_features**

*   Carregar o modelo
    *   model = pickle.load(open('model.pkl', 'rb'))
*   Inferir um nome
    *   model.predict(\['Carlo'\])\[0\] # saida “male”

Não utilizar com sobrenomes, o modelo foi treinado somente com os primeiros nomes de indivíduos

![](https://image_url.png)