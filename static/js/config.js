const globalConfig = {
    row: "cinq",
    time: {
        agent: 990,
        spy: 120,
        transition_start_agent: 8,
        transition_end_game: 15,
        transition_start_spy: 8,
    },
    index: {
        poeme_accueil: `<b class="FirstLetter">C</b>haque coup peut être le dernier</br><b class="FirstLetter">O</b>sez jouer, et aux meilleurs vous comparer</br><b class="FirstLetter">D</b>ans un classement sans aucune pitié</br><b class="FirstLetter">E</b>mbrassez les nouvelles technologies</br><b class="FirstLetter">N</b>'ayez pas peur de battre vos amis</br><b class="FirstLetter">A</b>vec notre IA, vous ne craignez rien</br><b class="FirstLetter">M</b>ais faites attention aux cartes noires</br><b class="FirstLetter">E</b>lles ne vous veulent pas du bien</br><b class="FirstLetter">S</b>oyez réfléchi pour ne pas décevoir</br>`,
    },
    rules: {
        titre_agent: "Manche 1 : Vous incarnez l'agent",
        txt_agent: "Vous devez deviner les mots que l'IA suggère ! <br><br>Durée: <b id='time_txt_rules_agent'></b>",
        titre_espion: "Manche 2 : Vous incarnez l'espion",
        txt_espion: "Vous devez faire deviner à l'IA certains mots ! <br><br>Durée: <b id='time_txt_rules_espion'></b>",
    },
    alert: {
        propal_IA: "Voici ma proposition :",
        propal_vocal_IA: "Veuiller cliquer sur le bouton pour faire une proposition.",
        propal_form_IA: "Faites une proposition",
        error_message: "Le mot que vous avez proposé n'est pas valide",
        error_messageIA: "Le mot auquel vous pensez n'a pas pu être deviné",
        error_messagePlateau: "Le mot que vous avez proposé ne fonctionne pas en raison d'un mot similaire/de la même famille sur le plateau"
    },
    cgu: {
        title_cgu: "Conditions d'utilisation",
        ul_cgu: "En utilisant ce service de récupération d'adresses e-mail, vous acceptez les conditions suivantes:",
        cgu1: "Nous ne partagerons pas vos informations personnelles avec des tiers sans votre consentement.",
        cgu2: "Nous utiliserons vos informations personnelles uniquement pour fournir le service de récupération d'adresses e-mail.",
        cgu3: "Nous prendrons des mesures raisonnables pour protéger vos informations personnelles contre la perte, l'utilisation abusive et l'accès non autorisé.",
        cgu4: "Nous ne conserverons pas vos informations personnelles plus longtemps que nécessaire pour fournir le service de récupération d'adresses e-mail.",
        cgu5: "Nous nous réservons le droit de modifier ces conditions d'utilisation à tout moment.",
        cgu_thx: "Merci d'avoir participé à notre jeu!"
    },
    transition: {
        espion: {
            infos: "Voici le prochain plateau à faire deviner à l'IA."
        }
    }
}