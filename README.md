# vacancy_detector

Nodos de deteccion de espacios libres para iniciar una accion `drop`.

- `table_drop_tf_node`: detecta una superficie de mesa, busca una zona libre y
  publica `table_drop_start`.
- `container_drop_tf_node`: detecta contenedores en piso, busca el centro del
  hueco y publica `container_drop_start`.
- `shelf_drop_tf_node`: detecta repisas horizontales, selecciona solo la
  seccion de estante mas cercana/enfrentada y publica `shelf_drop_start` en un
  punto libre dentro de esa seccion.