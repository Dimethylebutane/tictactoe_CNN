# tictactoe_CNN
> Petit projet de fin de semestre pour faire jouer un robot à câble au morpion. Ce projet a duré 3 jours et n'est pas fonctionnel.

L'idée est de placer un tableau blanc devant le robot. La caméra du robot voit le jeu et est donc capable de communiquer l'état de la partie à un algorithme (l'objet de ce repos) qui détermine le coup suivant (pas optimal pour laisser une chance à l'humain de gagner). La caméra sert aussi à mesurer la position du robot pour le contrôler en boucle fermé. Une partie mécanique permet de porter le feutre et de le faire appuyer ou non sur le tableau. Ces codes ne s'occupent que de la lecture du jeu avec la caméra. Il n'est pas fonctionnel du tout, en l'état actuel le programme ne permet même pas de déterminer la position du jeu de manière fiable.

Problèmes rencontrés:
- Le tableau est brillant, reflets qui nuisent à la qualité de l'image (contraste);
- La couleur des traits dépendent de l'état des feutres et de la vitesse d'écriture sur le tableau (contraste);
- Le tableau est tout blanc, l'autofocus de la webcam est perdu (flou)
    |> solution: filmer en partie autour du tableau

Fonctionnement:
- Un CNN trouve l'emplacement du jeu sur l'image.Ne fonctionne vraiment pas bien. C'est ma deuxième fois avec le Deep Learning, donc je ne maitrise pas grand-chose. Je pense que passer sur un algorithme classique (avec opencv et détection d'angle, ligne, ...) serait plus pertinent. L'interprétation de l'image pour sortir l'état du jeu n'est pas fait (manque de temps). le reste du projet était réalisé par mes camarades.

Perspectives d'évolution:
- mieux contrôler la qualité de l'image:
    |> focus constant
    |> limiter les reflets (placement judicieux de la caméra et du tableau dans le robot)
    |> utiliser des feutres neufs 

TODO:
- Refaire l'algorithme de détection du jeu (ROI (Region Of Interest) = le jeu seulement, ignorer le reste)- Faire un algorithme de lecture du jeu (en partant de la ROI, voir https://github.com/tempdata73/tic-tac-toe​)

Sources:
- https://github.com/tempdata73/tic-tac-toe​
- https://medium.com/analytics-vidhya/object-localization-using-keras-d78d6810d0be
