import numpy as np

#Method adapted from supp mat of [Mary C Meyer. Inference using shape-restricted regression splines. 2008]
def mspl(x, ys, w, k): #Monotone quadratic splines
  x = np.array(x).squeeze()
  ys = np.array(ys).squeeze()
    
  n = len(x)
  sm = 0.00000001
  q1 = np.ones(n) / np.sqrt(n)
  q = q1.reshape(-1,1)
  l = 1
  h = [1]
  r = 1

  knots = np.round(np.arange(k + 2) * n / (k + 1)).astype(int)
  knots[0] = 1
  t = x[knots-1]
  m = k + 3
  sigma = np.vstack([np.arange(1, m+1)*n for i in range(n)]).T
  sigma[0, :] = 1
  sigma = sigma.astype(float)

  for j in range(1, k):
      for i in range(1, knots[j-1]+1):
        sigma[j, i-1] = 0

      for i in range(knots[j-1]+1, knots[j]+1):
        sigma[j, i-1] = (x[i-1] - t[j-1]) ** 2 / (t[j + 1] - t[j-1]) / (t[j] - t[j-1])

      for i in range(knots[j]+1, knots[j+1]+1):
        sigma[j, i-1] = 1-(x[i-1]-t[j+1])**2/(t[j+1]-t[j])/(t[j+1]-t[j-1])

      for i in range(knots[j+1]+1, n+1):
        sigma[j, i-1] = 1

  ###
  j=k
  for i in range(1, knots[j-1]+1):
    sigma[j, i-1] = 0
  for i in range(knots[j-1]+1, knots[j]+1):
    sigma[j, i-1] = (x[i-1] - t[j-1]) ** 2 / (t[j + 1] - t[j-1]) / (t[j] - t[j-1])
  for i in range(knots[j]+1, knots[j+1]+1):
    sigma[j, i-1] = 1-(x[i-1]-t[j+1])**2/(t[j+1]-t[j])/(t[j+1]-t[j-1])

  ###
  for i in range(1, knots[1]+1):
    sigma[j+1, i-1] = 1-(t[1]-x[i-1])**2/(t[1]-t[0])**2
  for i in range(knots[1]+1, n+1):
    sigma[j+1, i-1]=1
  for i in range(1, knots[k]+1):
    sigma[j+2, i-1] = 0
  for i in range(knots[j]+1, knots[j+1]+1):
    sigma[j+2, i-1] = (x[i-1]-t[j])**2/(t[j+1]-t[j])**2

  ###
  id = np.eye(n)
  proj = id - q@q.T
  sigma = proj@sigma.T
  sigma = sigma.T
  sigma[0] = q1.squeeze()
  sigma0 = sigma

  #weights
  wmat = np.diag(np.sqrt(1 / w))
  winv = np.diag(np.sqrt(w))
  sigma = np.dot(sigma0, wmat)
  y = np.dot(wmat, ys)
  q1 = np.dot(wmat, q1)
  q1 = q1 / np.sqrt(np.sum(q1**2))
  q = q1.reshape(-1, 1)

  # Add one edge to start
  check = 0
  rhat = y - q @ q.T @ y
  b2 = sigma @ rhat

  if np.max(b2) > sm:
      obs = np.arange(1, m+1)
      i = obs[np.argmax(b2)]
      l += 1
      qnew = sigma[i-1, :] - q @ q.T @ sigma[i-1, :]
      qnew /= np.sqrt(np.sum(qnew**2))
      q = np.column_stack((q, qnew))
      h = np.append(h, i)
      r = q.T @ sigma[h-1, :].T

  if b2[np.argsort(b2)[-1]] < sm:  # Equivalent to R's rank
      check = 1

  # LOOP starts here:
  while check == 0:
    # Fit data to current EDGES
    a = q.T @ y

    # Check if convex:
    # First find the b vector
    b = np.zeros(l)

    b[l-1] = a[l-1] / r[l-1, l-1]
    for j in range(l-1, 2):
        b[j-1] = a[j-1]
        for i in range(j+1, l+1):
            b[j-1] -= r[j-1, i-1] * b[i-1]
        b[j-1] /= r[j-1, j-1]

    # Check to see if b positive
    obs = np.arange(2, l+1)
    i = obs[np.argsort(b[1:l])[0]]
    if b[i-1] < (-sm):
        # If not, remove hinge, make new q and r
        c1 = 0
        h = np.concatenate([h[0:(i-1)], h[i:l]])
        l -= 1
        q = q[:, :i-1]
        for j in range(i, l+1):
            qnew = sigma[h[j-1], :] - q @ q.T @ sigma[h[j-1]-1, :]
            qnew /= np.sqrt(np.sum(qnew**2))
            q = np.column_stack((q, qnew))
        r = q.T @ sigma[h-1, :].T

    if b[i-1] > (-sm):
        c1 = 1

        # Now see if we need to add another hinge
        theta = q @ q.T @ y
        rhat = y - theta
        b2 = sigma @ rhat

        # Check to see if b2 is negative
        obs = np.arange(1,m+1)
        i = obs[np.argmax(b2)]
        if b2[i-1] > sm:
            l += 1
            qnew = sigma[i-1, :] - q @ q.T @ sigma[i-1, :]
            qnew /= np.sqrt(np.sum(qnew**2))
            q = np.column_stack((q, qnew))
            h = np.append(h, i)
            r = q.T @ sigma[h-1, :].T
            c2 = 0

        if b2[i-1] < sm:
            c2 = 1
        check = c1 * c2

  yhat = np.dot(np.dot(winv, q), a)
  return yhat