ES_ENCODING = "latin-1"
EN_ENCODING = "latin-1"
PT_ENCODING = "utf-8"

OUTPUT_DIR = "results/pt_final_run"

SUBS_LABELS = ["gold_subs", "cand_subs"]
HEADER = ["subset", "mode", "ft-mode", "dataset", "prompt_1", "prompt_2", "model", "template", "k", "top_train",
          "MAP@1/Potential@1/Precision@1", "MAP@3", "MAP@5",
          "MAP@10", "Potential@3", "Potential@5", "Potential@10", "Accuracy@1@top_gold_1", "Accuracy@2@top_gold_1",
          "Accuracy@3@top_gold_1"]

SPANISH_DICT_PATH = "data/spanish_all.json"

DATA_DIR_EN = "data/en"
DATA_DIR_ES = "data/es"
DATA_DIR_PT = "data/pt"
DATA_DIR_ZH = "data/zh"

BENCH_LS_PATH = f"{DATA_DIR_EN}/raw/BenchLS/BenchLS.txt"
BENCH_LS_PARSED_PATH = f"{DATA_DIR_EN}/parsed/benchls.tsv"

NNS_EVAL_PATH = f"{DATA_DIR_EN}/raw/NNSeval/NNSeval.txt"
NNS_EVAL_PARSED_PATH = f"{DATA_DIR_EN}/parsed/nnseval.tsv"

LEX_MTURK_PATH = f"{DATA_DIR_EN}/raw/lex.mturk.14/lex.mturk.txt"
LEX_MTURK_PARSED_PATH = f"{DATA_DIR_EN}/parsed/lex.mturk.tsv"

CEFR_PATH = f"{DATA_DIR_EN}/raw/lex-simplification/open_dataset.tsv"
CEFR_PARSED_PATH = f"{DATA_DIR_EN}/parsed/open_dataset.cefr.tsv"

EASIER_PATH = f"{DATA_DIR_ES}/raw/EASIER_CORPUS/SGSS.csv"
EASIER_PARSED_PATH = f"{DATA_DIR_ES}/parsed/sgss.tsv"

SIMPLEX_PATH = f"{DATA_DIR_PT}/raw/SIMPLEX-PB-3.0/Simplex 3.0.xlsx"
SIMPLEX_PARSED_PATH = f"{DATA_DIR_PT}/parsed/simplex_3.0.csv.tsv"

LS_CHINESE_PATH = f"{DATA_DIR_ZH}/raw/Chinese-LS/annotation_data.csv"

EN_PARSED_FILES = [BENCH_LS_PARSED_PATH, NNS_EVAL_PARSED_PATH, LEX_MTURK_PARSED_PATH, CEFR_PARSED_PATH]
ES_PARSED_FILES = [EASIER_PARSED_PATH]

PT_PARSED_FILES = [SIMPLEX_PARSED_PATH]

LEXICAL_CHANGES = {
    "benchls.tsv": {
        "Hurricane-force wind gusts were reported in New England .": {"hurricane-force": "Hurricane-force"},
        "Galls indeed arise from the stinging of the plant tissues by the ovipositors of female gall wasps , "
        "and the egg laid in the plant tissues develops inside the gall into a grub , which eventually emerges "
        "full-grown and transformed into a mature gall wasp .": {
            "galls": "Galls"}
    },
    "lex.mturk.tsv": {
        "Companies may also take out strike insurance prior to an anticipated strike , to help offset the losses "
        "which the strike would cause .": {
            "companies": "Companies"},
        "Situated 13 miles southeast of Seattle , Washington , Renton straddles the southeast "
        "shore of Lake Washington .": {
            "situated": "Situated"},
        "Hurricane-force wind gusts were reported in New England .": {"hurricane-force": "Hurricane-force"},
        "Nevertheless , until the middle of the 20th century , agriculture dominated the canton .": {
            "nevertheless": "Nevertheless"},
    },
    "open_dataset.cefr.tsv": {
        "Identify the type of government in the United States and compare it to other forms of government ": {
            "identify": "Identify"},
        "In socialist countries , the government also usually owns and controls utilities such as electricity , "
        "transportation systems like airlines and railroads , and telecommunications systems . ": {
            "utility": "utilities"},
        "But when the Silk Road , the long overland trading route from China to the Mediterranean , "
        "became costlier and more dangerous to travel , Europeans searched for a more efficient and inexpensive trade "
        "route over water , initiating the development of what we now call the Atlantic World . ": {
            "initiate": "initiating"},
        "With this agricultural revolution , and the more abundant and reliable food supplies it brought , populations"
        " grew and people were able to develop a more settled way of life , building permanent settlements . ": {
            "supply": "supplies"},
        "Weapons made of obsidian , jewelry crafted from jade , feathers woven into clothing and ornaments , and cacao"
        " beans that were whipped into a chocolate drink formed the basis of commerce . ": {
            "weave": "woven"},
        "Flourishing along the hot Gulf Coast of Mexico from about 1200 to about 400 BCE , the Olmec produced a number"
        " of major works of art , architecture , pottery , and sculpture . ": {
            "flourish": "Flourishing"},
        "Flourishing from roughly 2000 BCE to 900 CE in what is now Mexico , Belize , Honduras , and Guatemala , the "
        "Maya perfected the calendar and written language the Olmec had begun . ": {
            "flourish": "Flourishing"},
        "Manufacturers of canned carbonated drinks take samples to determine if a 16 ounce can contains 16 ounces of "
        "carbonated drink . ": {
            "manufacturer": "Manufacturers"},
        "Numerical variables take on values with equal units such as weight in pounds and time in hours . ": {
            "numerical": "Numerical"},
        "A hypothesis may become a verified theory . ": {"verify": "verified"},
        "Inductive reasoning involves formulating generalizations inferred from careful observation and the analysis "
        "of a large amount of data . ": {
            "formulate": "formulating"},
        "Subsequently , they began to separate and use specific components of matter . ": {
            "subsequently": "Subsequently"},
        "Dyes , such as indigo and Tyrian purple , were extracted from plant and animal matter . ": {"dye": "Dyes"},
        "Subsequently , an amalgamation of chemical technologies and philosophical speculations were spread from "
        "Egypt , China , and the eastern Mediterranean by alchemists , who endeavored to transform `` base metals '' "
        "such as lead into `` noble metals '' like gold , and to create elixirs to cure disease and extend life . ": {
            "subsequently": "Subsequently"},
        "Chemical engineering , materials science , and nanotechnology combine chemical principles and empirical "
        "findings to produce useful substances , ranging from gasoline to fabrics to electronics . ": {
            "range": "ranging"},
        "Scarcity means that human wants for goods , services and resources exceed what is available . ": {
            "scarcity": "Scarcity"},
        "Virtually every major problem facing the world today , from global warming , to world poverty , to the "
        "conflicts in Syria , Afghanistan , and Somalia , has an economic dimension . ": {
            "virtually": "Virtually"},
        "Division and specialization of labor only work when individuals can purchase what they do not produce in "
        "markets . ": {
            "division": "Division"},
        "MERITS OF AN EDUCATION IN PSYCHOLOGY ": {"merit": "MERITS"},
        "They learn about basic principles that guide how we think and behave , and they come to recognize the "
        "tremendous diversity that exists across individuals and across cultural boundaries -LRB- American "
        "Psychological Association , 2011 -RRB- . ": {
            "boundary": "boundaries"},
        "Statistics from the United States Department of Agriculture show a complex picture . ": {
            "statistics": "Statistics"},
    },
    "sgss.tsv": {
        "La ministra ha explicado que Espa??a apuesta por producir salud mediante Estrategias Nacionales dirigidas a la poblaci??n general que pretenden q, en primer lugar, la introducci??n de un enfoque de salud en todas las pol??ticas; y en segundo, incluir medidas relacionas con el ciclo vital cuando es necesario (Alzheimer, Envejecimiento Activo, la lucha contra la Soledad no deseada o el programa de detecci??n de la fragilidad y actividad f??sica).": {
            "Enfoque": "enfoque"},
        "Es el caso de enfermedades renales (ri????n), hep??ticas (h??gado) cardiopat??as (coraz??n), fibrosis qu??stica (pulmones), enfermedad de Crohn y enfermedades metab??licas (aparato digestivo); Linfedema (sistema linf??tico), hemofilia (coagulaci??n de la sangre), lupus (sistema inmune); enfermedades reum??ticas (aparato locomotor); y cefaleas, migra??as, alzh??imer, p??rkinson, trastornos del sue??o, fibromialgia o s??ndrome de fatiga cr??nica (sistema nervioso central).": {
            "Cefaleas": "cefaleas"},
        "Todos los expertos coinciden en que el punto de arranque para prevenir y paliar los malos tratos a las personas mayores comienza con la tarea de informaci??n y formaci??n a toda la sociedad en su conjunto y a los profesionales que trabajan con las Personas Mayores en particular.": {
            "Prevenir": "prevenir"},
        "Estrategia integral": {"estrategia": "Estrategia"},
        "Favorecer un buen lavado bucal, tras la comida, para evitar posibles aspiraciones por restos de alimento o infecciones.": {
            "Tras": "tras"},
        "Consciente de los mencionados riesgos, la Agencia Espa??ola de Medicamentos y Productos Sanitarios (AEMPS), adscrita al Ministerio de Sanidad, Consumo y Bienestar Social, insiste en un dec??logo de recomendaciones generales que busca promover una exposici??n solar segura y fomentar el buen uso de los cosm??ticos:": {
            "consciente": "Consciente"},
        "???Conocemos que el mayor nivel educativo de las familias, en particular de las madres, el incremento de los ingresos del hogar o el simple hecho de que las familias coman juntas, son factores que mejoran la salud de sus miembros???, ha explicado.": {
            "En particular": "en particular"},
        "Esta es una gran aportaci??n que incorpora la Agenda 2030 y una gran aportaci??n para las personas con discapacidad, teniendo en cuenta que lo que diferencia a las personas en desigualdad social con las personas con discapacidad es que las primeras pueden cambiar su situaci??n pero las personas con discapacidad no la van a cambiar, entonces habr??a que remover barreras.": {
            "Incorpora": "incorpora"},
        "Sanidad recuerda las recomendaciones para prevenir los da??os derivados de las altas temperaturas": {
            "Prevenir": "prevenir"},
        "El Plan que ha aprobado el Consejo Interterritorial incorpora medidas como la inclusi??n de diagn??stico de la infecci??n en colectivos singularizados y en situaciones cl??nicas determinadas": {
            "Incorpora": "incorpora"},
        "Tambi??n abandona la concepci??n m??dico-rehabilitadora y asistencial y la sustituye por el enfoque ???convencionalista???, basado en el reconocimiento de los derechos y deberes de las personas con discapacidad.": {
            "Enfoque": "enfoque"},
        "El tercer objetivo general planteado en el documento, informar de los problemas de suministro, tiene como finalidad satisfacer, en la medida de lo posible, las expectativas de pacientes y profesionales sanitarios proporcion??ndoles informaci??n de m??xima utilidad de la forma m??s ??gil.": {
            "Satisfacer": "satisfacer"},
        "La primera jornada se desarrollar?? en sesi??n de ma??ana y tarde y, previo a la inauguraci??n, tendr?? lugar una ponencia sobre la evoluci??n de la atenci??n a la dependencia en Castilla-La Mancha a cargo de la consejera de Bienestar Social.": {
            "Jornada": "jornada"},
        "Evoluci??n de la dependencia": {"evoluci??n": "Evoluci??n"},
        "A su juicio, ???tenemos que ser capaces de impulsar un cambio cultural dirigido al desarrollo pleno de la autonom??a personal en todo el ciclo vital y que este derecho est?? garantizado por las administraciones y los ??rganos de los que depende, bas??ndose en el respeto a la dignidad de todas las personas y la igualdad de oportunidades???.": {
            "a su juicio": "A su juicio"},
        "Conmemoraci??n del V centenario de la muerte del Bosco": {"conmemoraci??n": "Conmemoraci??n"},
        "Junto a estas obras, los pr??stamos procedentes de Lisboa, Londres, Berl??n, Viena, Venecia, Rotterdam, Par??s, Nueva York, Filadelfia o Washington, entre otras ciudades, hacen de esta muestra un acontecimiento ??nico para sumergirse en el imaginario de uno de los pintores m??s fascinantes del arte universal.": {
            "Muestra": "muestra"},
        "Si todo va bien es posible que consigas un hito hist??rico y mundial de hollar 14 ocho miles con 77 a??os???": {
            "Hollar": "hollar"},
        "Consolidado a d??a de hoy como programa estable  del MUBAM (Museo de Bellas Artes de Murcia)  est?? abierto a todos los pacientes, familiares y cuidadores que lo soliciten.": {
            "consolidado": "Consolidado"},
        "Aglutina todos los mecanismos que conoce la psicolog??a moderna para transformarnos": {"aglutina": "Aglutina"},
        "Y, posteriormente, inform?? in situ de las sucesivas crisis de Irak y del conflicto palestino-israel??: un largu??simo periplo laboral, por el que ???encabeza??? el r??nking de las primeras espadas del periodismo actual en nuestro pa??s, considerada por sus colegas un referente mundial en temas de Oriente Me- dio": {
            "ranking": "r??nking"},
        "??Sabes que cada d??a que pasa ganamos seis horas de vida y que por cada a??o transcurrido podemos vivir tres meses m??s que el anterior": {
            "Transcurrido": "transcurrido"},
        "Premisas de la silver economy": {"Silver economy": "silver economy"},
        "Promover y crear un di??logo constructivo y duradero entre los representantes de las organizaciones de discapacidad y las organizaciones de las personas mayores con los sectores interesados, tales como autoridades y promotores, profesionales, fabricantes, etc.": {
            "promover": "Promover"},
        "Especialista en Trastornos del Sue??o e Hipnosis, por la Universidad Complutense de Madrid, tambi??n ha realizado un m??ster en Gerontolog??a y Salud por la Universidad de Alicante.": {
            "trastornos": "Trastornos"},
        "Promovido por el Ministerio de Sanidad, Consumo y Bienestar Social junto con el Ministerio de Agricultura, Pesca y Alimentaci??n y coordinado por la AEMPS con la participaci??n de todas las comunidades aut??nomas y ocho ministerios (Sanidad, Agricultura, Econom??a, Interior, Defensa, Educaci??n, Ciencia y Transici??n Ecol??gica), el PRAN trabaja desde 2014 con el objetivo de frenar el desarrollo y la diseminaci??n de la resistencia bacteriana": {
            "promovido": "Promovido"},
        "Tambi??n en el marco del D??a Europeo para el Uso Prudente de los Antibi??ticos y la Semana Mundial de Concienciaci??n sobre el Uso de los Antibi??ticos, el PRAN ha organizado la I Carrera Popular Universitaria ?????Corre sin resistencias!??? junto a las facultades biosanitarias de las universidades de Alcal?? de Henares, Almer??a, Bilbao, Murcia y Sevilla.": {
            "prudente": "Prudente"},
        "Asimismo, se celebrar?? una nueva edici??n de la Jornada del D??a Europeo para el Uso Prudente de los Antibi??ticos el 18 de noviembre en la sede del Ministerio de Sanidad, Consumo y Bienestar Social": {
            "prudente": "Prudente"},
        "??lceras o llagas": {"??lceras": "??lceras"},
        "Carmen Su??rez, jefa del servicio de medicina Interna del Hospital de La Princesa de Madrid, ha resaltado que ???la mejora en la prevenci??n, la detecci??n m??s precoz de los factores de riesgo y la definici??n de procesos asistenciales locales que sirvan para asegurar el ??ptimo seguimiento y atenci??n del paciente vascular???, ser??an los retos a los que se deben enfrentar todos los actores involucrados en el tratamiento y seguimiento del paciente vascular.": {
            "Paciente": "paciente"},
        "Seg??n Rodrigo Radovan, director del ??rea Movilidad en Espa??a de T??V Rheinland, ???es muy importante poner en pr??ctica estos consejos para prevenir accidentes y evitar situaciones de riesgo, asegurando as?? poder disfrutar al m??ximo del merecido descanso???.": {
            "Prevenir": "prevenir"},
        "Chequear los neum??ticos": {"chequear": "Chequear"},
        "Carles Gaig Ventura, Coordinador del Grupo de Estudio de Trastornos de la Vigilia y Sue??o de la Sociedad Espa??ola de Neurolog??a (SEN)": {
            "trastornos": "Trastornos"},
        "Para profundizar en esta nueva realidad se ha llevado a cabo el simposio ???Cerca del Paciente Mayor???, organizado por Janssen y celebrado en el marco del V Workshop del Grupo Espa??ol de Hematogeriatr??a (GEHEG) de la Sociedad Espa??ola de Hematolog??a y Hemoterapia (SEHH) en el Hospital de la Santa Creu i Sant Pau, de Barcelona.": {
            "workshop": "Workshop"},
        "La diversidad de pa??ses y regiones que forman este Consorcio, con diferentes sistemas y pol??ticas socio-sanitarias y contextos culturales, sociales y econ??micos dispares, representan un formidable reto para conseguirlo pero tambi??n una oportunidad para, aunando esfuerzos promover una pol??tica Europea efectiva y competente para prevenir y tratar la fragilidad y la discapacidad ligada a ella.": {
            "consorcio": "Consorcio"},
        "La di??lisis peritoneal es una modalidad de tratamiento renal sustitutivo (TRS) que cada paciente realiza en su propio domicilio, sin necesidad de acudir a un centro m??dico, y se basa fundamentalmente en que la di??lisis se realiza a trav??s de una membrana del propio individuo, el peritoneo.": {
            "Acudir": "acudir"},
        "Por esta raz??n, desde el COEM promovemos este tipo de campa??as, con las que tratamos de trasladar al ciudadano la importancia del mensaje y la necesidad de acudir al dentista cada seis mes para cuidar en la prevenci??n???, afirma el Dr": {
            "Acudir": "acudir"},
        "Incrementar el consumo de frutas y verduras a 5 raciones al d??a.": {"incrementar": "Incrementar"},
        "Estos son algunos de los datos que recoge el primer Libro Blanco de la Nutrici??n Infantil en Espa??a, que se ha presentado el 26 de noviembre en Barcelona": {
            "nutrici??n": "Nutrici??n"},
        "Primer Libro Blanco de la Nutrici??n Infantil Espa??ola": {"nutrici??n": "Nutrici??n"},
        "La presentaci??n se enmarca dentro de la III Jornada Cient??fica de la C??tedra Ordesa de Nutrici??n Infantil, creada por Laboratorios Ordesa y ubicada en la Universidad de Zaragoza con la participaci??n tambi??n de la Universidad de Cantabria.": {
            "nutrici??n": "Nutrici??n"},
        "Junto a ellos, ex jugadores de f??tbol y residentes,  compartir??n con los asistentes su experiencia en estos talleres de reminiscencia basados en los recuerdos del f??tbol.": {
            "Reminiscencia": "reminiscencia"},
        "Comer sano y bien no est?? re??ido???, afirma Adolfo Mu??oz, Premio Nacional de Gastronom??a y embajador en su restaurante Palacio de Cibeles de la presentaci??n en Madrid del recetario del cual se editar??n 2.500 recetarios que se distribuir??n gratuitamente durante la Semana del Coraz??n de Madrid": {
            "gastronom??a": "Gastronom??a"},
        "Jos?? M?? Ordov??s, Director del laboratorio de Nutrici??n y Gen??mica del Human Nutrition Research Center on Aging de la Universidad de Tufts y profesor de Nutrici??n y Gen??tica en la Sackler School of Biomedical Sciences, Boston (EEUU), indic?? en su exposici??n ???Gen??tica y Nutrici??n??? que ???hubo un momento en que la predisposici??n a engordar era posiblemente protectora porque facilitaba la supervivencia en ??pocas de escasez, algo que debi?? abundar en el pasado": {
            "nutrici??n": "Nutrici??n"},
        "Asociaci??n de Servicio Integral Sectorial para Ancianos (ASISPA)": {"integral": "Integral"},
        "El CPAP de Bergondo (A Coru??a) particip?? en la jornada formativa para la obtenci??n de la Licencia de Navegaci??n organizada por Predif Galicia para personas con discapacidad del Centro de Bergondo, gracias a la financiaci??n de Fundaci??n ONCE": {
            "licencia": "Licencia"},
        "A su juicio, el ingreso tambi??n es una cuesti??n de seguridad, porque los ciudadanos tienen que tener unos m??nimos niveles de seguridad material que les permitan satisfacer sus necesidades b??sicas, y una cuesti??n de libertad, porque no hay libertad si una persona tiene que invertir toda su energ??a en sobrevivir.": {
            "a su juicio": "A su juicio"},
        "Bas??ndose en m??s de un siglo de investigaci??n e innovaci??n, Nutricia ha aprovechado el poder de la nutrici??n que salva y cambia vidas, para crear un portfolio pionero en nutrici??n especializada que puede cambiar la trayectoria de la salud a lo largo de la vida.": {
            "Pionero": "pionero"},
        "Donar Km": {"km": "Km"},
        "??Libro Blanco de la Nutrici??n de las Personas Mayores??": {"nutrici??n": "Nutrici??n"},
        "Otras alternativas interesantes son el t?? y las infusiones, a??aden los autores del Libro Blanco de la Nutrici??n de las personas mayores en Espa??a.": {
            "nutrici??n": "Nutrici??n"},
        "Bel??n Gal??n, directora de Marketing y Comunicaci??n de thyssenkrupp Home Solutions ha dicho: ???La crisis del coronavirus, el encierro en casa, el aislamiento de los seres queridos y del resto de la gente y los malos datos que daban cada d??a por la televisi??n han hecho que muchos de nuestros mayores hayan pasado gran parte del confinamiento con tristeza y a??oranza": {
            "marketing": "Marketing"},
        "Se trata de una actividad grupal y psicodin??mica encaminada a la orientaci??n espacial y temporal, la estimulaci??n de capacidades cognitivas, la interacci??n social y la reminiscencia de momentos emotivos.": {
            "Reminiscencia": "reminiscencia"},
        "Asimismo, apela a que, al margen de esta mesa de negociaci??n, cuyos trabajos en principio durar??n hasta comienzos de verano, tengan continuidad posteriormente los contactos y se siga avanzando para buscar un pacto de Estado en sanidad de manera estructural y permanente, m??s all?? de esta pandemia.": {
            "Posteriormente": "posteriormente"},
        "Incide en que es clave acabar no solo con el 21% de IVA de materiales y productos, sino de servicios como el transporte de sangre o la seguridad y la limpieza en centros hospitalarios, que siguen estando gravados al 21%.": {
            "incide": "Incide"},
        "Canal S??nior es una entidad no lucrativa, de ??mbito nacional, especializada en potenciar el conocimiento del colectivo senior, a trav??s de la utilizaci??n de internet y las nuevas tecnolog??as, posibilitando que los ciudadanos, en general, y m??s en concreto los del entorno rural, tengan el conocimiento a un clic de distancia": {
            "s??nior": "S??nior"},
        "Dentro de esta iniciativa y con motivo del D??a de la Solidaridad Intergeneracional, la Plataforma Canal S??nior han recopilado una gran variedad de cursos gratuitos, con el objetivo de que cualquier persona mayor pueda aprender durante esta cuarentena para estar m??s conectada con su entorno, a trav??s de cursos online que pueden realizarse en cualquier momento sobre WhatsApp, cursos de consumo o cursos orientados a aprender a comprar.": {
            "intergeneracional": "Intergeneracional"},
        "Empoderar a las personas mayores en todos los ??mbitos del desarrollo, incluida su participaci??n en la vida social, econ??mica y pol??tica, ayuda tanto a garantizar su inclusi??n como a reducir las numerosas desigualdades a las que nos enfrentamos muchas personas.": {
            "empoderar": "Empoderar"},
        "Pere Clav??, director de Investigaci??n del Hospital de Matar??, Consorci Sanitari del Maresme (Barcelona) y Presidente Fundador de la Sociedad Europea de Trastornos de Degluci??n": {
            "degluci??n": "Degluci??n"},
        "Peres Clav??, director de Investigaci??n del Hospital de Matar??, Consorci Sanitari del Maresme (Barcelona) y Presidente Fundador de la Sociedad Europea de Trastornos de Degluci??n": {
            "trastornos": "Trastornos"},
    }
}
