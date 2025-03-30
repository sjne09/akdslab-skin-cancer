<div>
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UNI</th>
      <th>Prov-GigaPath</th>
      <th>PRISM/virchow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>architecture</th>
      <td>ViT large, patch size 16</td>
      <td>ViT giant, patch size 14</td>
      <td>ViT huge, patch size 14</td>
    </tr>
    <tr>
      <th>params</th>
      <td>303,350,784</td>
      <td>1,134,953,984</td>
      <td>631,229,184</td>
    </tr>
    <tr>
      <th>model_size</th>
      <td>1157.19MB</td>
      <td>4329.51MB</td>
      <td>2407.95MB</td>
    </tr>
    <tr>
      <th>runtime</th>
      <td>11.8401 sec/k tiles</td>
      <td>7.3335 sec/k tiles</td>
      <td>7.2343 sec/k tiles</td>
    </tr>
    <tr>
      <th>embed_dim</th>
      <td>1024</td>
      <td>1536</td>
      <td>2560</td>
    </tr>
  </tbody>
</table>
</div>