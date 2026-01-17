import { Route, RouterProvider, Redirect, useRouter } from './router/router';
import HomePage from './pages/HomePage';
import WorkspacePage from './pages/WorkspacePage';

function Routes() {
  const { pathname } = useRouter();

  if (pathname !== '/' && pathname !== '/workspace') {
    return <Redirect to="/" />;
  }

  return (
    <>
      <Route path="/">
        <HomePage />
      </Route>
      <Route path="/workspace">
        <WorkspacePage />
      </Route>
    </>
  );
}

export default function AppRouter() {
  return (
    <RouterProvider>
      <Routes />
    </RouterProvider>
  );
}
