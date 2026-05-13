import type { GetServerSideProps, NextPage } from "next";

export const getServerSideProps: GetServerSideProps = async () => ({
  redirect: { destination: "/analyze", permanent: false },
});

const Home: NextPage = () => null;

export default Home;
